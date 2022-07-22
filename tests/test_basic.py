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


# change this to false if it starts causing problems, but it should be stable:
use_multiprocessing = True

file_search_string = 'structural/*.unp'
mapping_coefficients = [0.0e-10,0.0e-6,0.0,0.0]
dispersion_coefficients = [-6.2e-09, -3.7e-05, 0.0, 0.0]

bit_shift_right = 4
dtype=np.uint16

# set fbg_position to None to skip fbg alignment
fbg_position = None
spectrum_start = 159
spectrum_end = 1459

fft_oversampling_size = 4096
bscan_z1 = 2900
bscan_z2 = -100
bscan_x1 = 0
bscan_x2 = -20

def process(filename,diagnostics=False,show_processed_data=True,manual_dispersion=False,manual_mapping=False,n_skip=0,do_rigid_registration=False,dispersion_coefficients=dispersion_coefficients):
    # setting diagnostics to True will plot/show a bunch of extra information to help
    # you understand why things don't look right, and then quit after the first loop
    # setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look

    # PARAMETERS FOR RAW DATA SOURCE
    cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

    n_vol = cfg['n_vol']
    n_slow = cfg['n_slow']
    n_repeats = cfg['n_bm_scans']
    n_fast = cfg['n_fast']
    n_depth = cfg['n_depth']

    # some conversions to comply with old conventions:
    n_slow = n_slow//n_repeats
    n_fast = n_fast*n_repeats


    # Make directories, as needed:
    output_directory_bscans = filename.replace('.unp','')+'_bscans'
    os.makedirs(output_directory_bscans,exist_ok=True)

    output_directory_info = filename.replace('.unp','')+'_info'
    os.makedirs(output_directory_info,exist_ok=True)

    if show_processed_data:
        output_directory_png = filename.replace('.unp','')+'_png'
        os.makedirs(output_directory_png,exist_ok=True)

    diagnostics_base = diagnostics
    diagnostics_directory = filename.replace('.unp','')+'_diagnostics'
    os.makedirs(diagnostics_directory,exist_ok=True)

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

    if show_processed_data:
        processing_fig = plt.figure(0,figsize=(4,6))

    for frame_index in range(n_slow):
        if diagnostics_base or frame_index==0:
            diagnostics = (diagnostics_directory,frame_index)
        else:
            diagnostics = False
            
        print('Frame %d of %d.'%(frame_index,n_slow))
        frame = src.get_frame(frame_index,diagnostics=diagnostics)
        frame = blob.dc_subtract(frame,diagnostics=diagnostics)
        frame = blob.k_resample(frame,mapping_coefficients,diagnostics=diagnostics)
        frame = blob.dispersion_compensate(frame,dispersion_coefficients,diagnostics=diagnostics)
        frame = blob.gaussian_window(frame,0.9,diagnostics=diagnostics)
        
        bscan_series = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)

        bscan = bscan_series
        bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)


        if show_processed_data:
            png_out_filename = os.path.join(output_directory_png,'bscan_%05d.png'%frame_index)
            
            plt.figure(0)

            plt.clf()
            plt.imshow(20*np.log10(np.abs(bscan)),aspect='auto',cmap='gray',clim=(40,90))
            plt.colorbar()
            plt.title('bscan dB')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(png_out_filename,dpi=150)
            plt.pause(.1)
            
        np.save(bscan_out_filename,bscan)
            
        if diagnostics_base:
            # use plt.close('all') instead of plt.show() if you want to save the diagnostic plots
            # without seeing them
            plt.close('all')
            #plt.show()

        if show_processed_data:
            plt.pause(.001)



def proc(fn):
    return process(fn,diagnostics=False,manual_dispersion=False,manual_mapping=False,n_skip=0)

if __name__=="__main__":

    flist = sorted(glob.glob(file_search_string))

    if use_multiprocessing:
        # parallelize the loop over files
        with mp.Pool(max(len(flist),4)) as p:
            p.map(proc,flist)
    else:
        # serialize the loop over files
        for unp_filename in flist:
            # it seems that for the Axsun, n_skip is always 0; can omit this step:
            #n_skip = identify_skip_frames(unp_filename,diagnostics=False)
            process(unp_filename,diagnostics=False,manual_dispersion=False,n_skip=0)
