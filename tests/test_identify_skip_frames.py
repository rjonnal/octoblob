from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader,dispersion_ui
from octoblob.bmp_tools import savebmp
import glob
import multiprocessing as mp
from octoblob.registration_tools import rigid_shift


# dispersion coefficients output by manual adjustment:
mapping_coefficients = [-5.598141695702673e-10, 7.486559139784956e-07, 0.0, 0.0]
dispersion_coefficients = [-7.676063773624743e-09, -2.8924731182795676e-05, 0.0, 0.0]

bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

fft_oversampling_size = 4096
bscan_z1 = 2900
bscan_z2 = -500
bscan_x1 = 0
bscan_x2 = -100

# parameters for bulk motion correction and phase variance calculation:
# original values:
# bulk_correction_threshold = 0.3
# phase_variance_threshold = 0.43

bulk_correction_threshold = 0.5
phase_variance_threshold = 0.5#0.43


def identify_skip_frames(filename,diagnostics=False):
    # This program looks at cross-correlation between consecutive frames to determine where the slow scanner has
    # moved, so that the repeated B-scans are clustered together correctly. Use it to set skip_frames correctly
    # PARAMETERS FOR RAW DATA SOURCE
    
    cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

    n_vol = cfg['n_vol']
    n_slow = cfg['n_slow']
    # we'll make one stack consisting of 10 groups of repeat scans:
    n_repeats = cfg['n_bm_scans']
    n_repeats_big_stack = n_repeats*10
    
    n_fast = cfg['n_fast']
    n_fast_original = n_fast
    n_depth = cfg['n_depth']

    # some conversions to comply with old conventions:
    n_slow = n_slow
    n_fast = n_fast*n_repeats_big_stack

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats_big_stack,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,dtype=dtype)

    frame = src.get_frame(0)
    frame = blob.dc_subtract(frame)
    frame = blob.k_resample(frame,mapping_coefficients)
    frame = blob.dispersion_compensate(frame,dispersion_coefficients)
    frame = blob.gaussian_window(frame,0.9)
    bscan_series = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2)
    stack_complex = blob.reshape_repeats(bscan_series,n_repeats_big_stack,x1=bscan_x1,x2=bscan_x2)

    def xcm(tar,ref):
        ftar = np.fft.fft2(tar)
        fref = np.conj(np.fft.fft2(ref))
        xc = np.abs(np.fft.ifft2(ftar*fref))
        return xc.max()

    def corr(tar,ref):
        atar = np.abs(tar)
        aref = np.abs(ref)
        sy,sx = tar.shape
        return np.sum((atar-atar.mean())*(aref-aref.mean()))/(np.std(atar)*np.std(aref))/float(sy)/float(sx)
        
    # check cross-correlation of frames to make sure that our n_skip was set correctly
    
    vals = []
    for idx1,idx2 in zip(range(stack_complex.shape[2]-1),range(1,stack_complex.shape[2])):
        #val = xcm(stack_complex[:,:,idx1],stack_complex[:,:,idx2])
        val = corr(stack_complex[:,:,idx1],stack_complex[:,:,idx2])
        vals.append(val)

    vals = np.array(vals)

    if diagnostics:
        plt.figure()
        plt.plot(vals)
        plt.xlabel('nth frame correlation with (n+1)th frame')
        plt.ylabel('correlation')
    
    vals_sums = []
    offsets = range(n_repeats)
    for offset in offsets:
        comb_indices = range(offset,len(vals),n_repeats)
        vals_sums.append(np.sum(vals[comb_indices]))

    if diagnostics:
        plt.figure()
        plt.plot((np.array(offsets)+1)%n_repeats,vals_sums,'ks')
        plt.show()

    # A bit of weird modulo math, necessitated by the
    # way the correlations were calculated:
    # the correlations (vals) are between the 0th and 1st
    # frame, then the 1st and 2nd frame, etc.
    # Therefore, we expect minima (low correlations)
    # at the n_skip-1, 2*n_skip-1, 3*n_skip-1, etc. locations
    n_skip_frames = (np.argmin(vals_sums)+1)%n_repeats

    # And, now since n_skip should be given in A-line index
    # rather than frame index. What a mess:
    return n_skip_frames*n_fast_original


def proc(fn):
    return process(fn,diagnostics=False,manual_dispersion=False,n_skip=0)

flist = sorted(glob.glob('angio/*.unp'))

# serialize the loop over files
for filename in flist:
    # it seems that for the Axsun, n_skip is always 0; can omit this step:
    n_skip = identify_skip_frames(filename,diagnostics=False)
    print(n_skip)
