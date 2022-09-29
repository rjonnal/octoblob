from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader,dispersion_tools
from octoblob.bmp_tools import savebmp
import glob
import multiprocessing as mp
from octoblob.registration_tools import rigid_shift


# # # quick rigid_shift test:
# source = np.random.randn(100,100)
# ref = source[2:52,3:53]
# tar = source[:50,:50]
# shifted_tar = rigid_shift(ref,tar)

# plt.figure()
# plt.imshow(ref)
# #plt.figure()
# #plt.imshow(tar)
# plt.figure()
# plt.imshow(shifted_tar)
# plt.show()

# sys.exit()

# dispersion coefficients output by manual adjustment:
mapping_coefficients = [-5.598141695702673e-10, 7.486559139784956e-07, 0.0, 0.0]
dispersion_coefficients = [-7.676063773624743e-09, -2.8924731182795676e-05, 0.0, 0.0]

# mapping and dispersion coefficients [m3,m2,c3,c2] output by test_optimize_mapping_dispersion_swept_source:
#mapping_coefficients = [2.71487217e-10,-3.74028591e-07,0.0,0.0]
#dispersion_coefficients = [-6.67475212e-09,-8.97711135e-06,0.0,0.0]

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

# setting diagnostics to True will plot/show a bunch of extra information to help
# you understand why things don't look right, and then quit after the first loop

# setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look

def process_unp(filename,diagnostics,show_processed_data=True,manual_dispersion=False,n_skip=0):

    # PARAMETERS FOR RAW DATA SOURCE
    cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

    output_directory_bscans = filename.replace('.unp','')+'_bscan_stacks'
    os.makedirs(output_directory_bscans,exist_ok=True)
    output_directory_angiograms = filename.replace('.unp','')+'_angiograms'
    os.makedirs(output_directory_angiograms,exist_ok=True)

    if show_processed_data:
        output_directory_png = filename.replace('.unp','')+'_png'
        os.makedirs(output_directory_png,exist_ok=True)

    n_vol = cfg['n_vol']
    n_slow = cfg['n_slow']
    n_repeats = cfg['n_bm_scans']
    n_fast = cfg['n_fast']
    n_depth = cfg['n_depth']

    # some conversions to comply with old conventions:
    n_slow = n_slow//n_repeats
    n_fast = n_fast*n_repeats

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

    if manual_dispersion:
        # check the dispersion coefficients
        # first we need a process function that takes a frame and the c3 and c2 coefficients
        # and returns a B-scan; we can copose this out of several blob functions:
        def process(frame,c3,c2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),mapping_coefficients),[c3,c2,0.0,0.0]),0.9))[800:1200,:]
        points,maxes = dispersion_tools.dispersion_tools(src.get_frame(0),process)

        c2,c3 = points[np.argmax(maxes)]
        print('Optimized coefficients:')
        print([c3,c2,0.0,0.0])
        sys.exit()

    if manual_mapping:
        # check the mapping coefficients
        # first we need a process function that takes a frame and the m3 and m2 coefficients
        # and returns a B-scan; we can copose this out of several blob functions:
        def process(frame,m3,m2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),[m3,m2,0.0,0.0]),dispersion_coefficients),0.9))[800:1200,:]
        
        points,maxes = dispersion_tools.dispersion_tools(src.get_frame(0),process,c3min=-1e-9,c3max=1e-9,c2min=-1e-6,c2max=1e-6)

        m2,m3 = points[np.argmax(maxes)]
        print('Optimized mapping coefficients:')
        print([m3,m2,0.0,0.0])
        sys.exit()
        
    if show_processed_data:
        processing_fig = plt.figure(0,figsize=(4,6))

    # In this section, we will load one set of repeats and arrange them in a 3D array
    # to be bulk-motion corrected

    for frame_index in range(n_slow):
        print(frame_index)
        frame = src.get_frame(frame_index)
        frame = blob.dc_subtract(frame)
        frame = blob.k_resample(frame,mapping_coefficients)
        frame = blob.dispersion_compensate(frame,dispersion_coefficients,diagnostics=diagnostics)
        frame = blob.gaussian_window(frame,0.9)
        bscan_series = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)
        stack_complex = blob.reshape_repeats(bscan_series,n_repeats,x1=bscan_x1,x2=bscan_x2)


        ref = stack_complex[:,:,0]
        for k in range(1,stack_complex.shape[2]):
            tar = stack_complex[:,:,k]
            stack_complex[:,:,k] = rigid_shift(ref,tar,max_shift=10,diagnostics=False)
        
        bscan = np.mean(np.abs(stack_complex),2)

        phase_variance = blob.make_angiogram(stack_complex,
                                             bulk_correction_threshold=bulk_correction_threshold,
                                             phase_variance_threshold=phase_variance_threshold,
                                             diagnostics=diagnostics)

        bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_stack_%05d.npy'%frame_index)
        angiogram_out_filename = os.path.join(output_directory_angiograms,'angiogram_bscan_%05d.npy'%frame_index)


        if show_processed_data:
            png_out_filename = os.path.join(output_directory_png,'bscan_angiogram_%05d.png'%frame_index)
            
            plt.figure(0)

            plt.clf()
            plt.subplot(2,1,1)
            plt.imshow(20*np.log10(np.abs(bscan)),aspect='auto',cmap='gray',clim=(40,90))
            plt.colorbar()
            plt.title('bscan dB')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,1,2)
            plt.imshow(phase_variance,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('angiogram (pv)')
            plt.xticks([])
            plt.yticks([])

            plt.savefig(png_out_filename,dpi=150)
            
            plt.pause(.1)

        # here we're saving the complex stack--could abs and average them first if we need to save disk space
        np.save(bscan_out_filename,stack_complex)

        np.save(angiogram_out_filename,phase_variance)

        if diagnostics:
            plt.show()


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
    return process_unp(fn,diagnostics=False,manual_dispersion=False,n_skip=0)

flist = sorted(glob.glob('angio/*.unp'))

# to do diagnostics, do something like the following:
# process_unp(flist[0],diagnostics=True)

# change this to false if it starts causing problems, but it should be stable:
use_multiprocessing = True

if use_multiprocessing:
    # parallelize the loop over files
    with mp.Pool(max(len(flist),4)) as p:
        p.map(proc,flist)
else:
    # serialize the loop over files
    for unp_filename in flist:
        # it seems that for the Axsun, n_skip is always 0; can omit this step:
        #n_skip = identify_skip_frames(unp_filename,diagnostics=False)
        process_unp(unp_filename,diagnostics=False,manual_dispersion=False,n_skip=0)
