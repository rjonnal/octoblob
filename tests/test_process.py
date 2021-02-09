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


test_type = 'angio'

if test_type=='structural':
    file_search_string = 'structural/*.unp'
    mapping_coefficients = [0.0e-10,0.0e-6,0.0,0.0]
    dispersion_coefficients = [-6.2e-09, -3.7e-05, 0.0, 0.0]
if test_type=='angio':
    # coefficients output by manual adjustment,
    # for the angiographic test set:
    file_search_string = 'angio/*.unp'
    mapping_coefficients = [3.2e-10,-1e-6,0.0,0.0]
    dispersion_coefficients = [3.6e-9,-7.1e-7,0.0,0.0]


bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

fft_oversampling_size = 4096
bscan_z1 = 2900
bscan_z2 = -100
bscan_x1 = 0
bscan_x2 = -100

# parameters for bulk motion correction and phase variance calculation:
# original values:
# bulk_correction_threshold = 0.3
# phase_variance_threshold = 0.43

bulk_correction_threshold = 0.5
phase_variance_threshold = 0.5#0.43

def process(filename,diagnostics=False,show_processed_data=True,manual_dispersion=False,manual_mapping=False,n_skip=0,do_angiography=None,do_rigid_registration=False,dispersion_coefficients=dispersion_coefficients):
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

    # caller can specify do_angiography to force, but if it is None (default), use n_repeats to decide
    if do_angiography is None:
        do_angiography = n_repeats > 1

    # Make directories, as needed:
    
    output_directory_bscans = filename.replace('.unp','')+'_bscans'
    os.makedirs(output_directory_bscans,exist_ok=True)

    output_directory_info = filename.replace('.unp','')+'_info'
    os.makedirs(output_directory_info,exist_ok=True)

    if do_angiography:
        output_directory_angiograms = filename.replace('.unp','')+'_angiograms'
        os.makedirs(output_directory_angiograms,exist_ok=True)

    if show_processed_data:
        output_directory_png = filename.replace('.unp','')+'_png'
        os.makedirs(output_directory_png,exist_ok=True)

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

    if manual_dispersion:
        # check the dispersion coefficients
        # first we need a process function that takes a frame and the c3 and c2 coefficients
        # and returns a B-scan; we can copose this out of several blob functions:
        def process_for_ui(frame,c3,c2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),mapping_coefficients),[c3,c2,0.0,0.0]),0.9),oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)
        #[800:1200,:]
        points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process_for_ui)

        # c2,c3 = points[np.argmax(maxes)]
        c2,c3 = points[-1]

        dispersion_coefficients = [c3,c2,0.0,0.0]
        np.savetxt(os.path.join(output_directory_info,'dispersion_coefficients.txt'),dispersion_coefficients)
        print('Optimized dispersion coefficients:')
        print([c3,c2,0.0,0.0])

    if manual_mapping:
        # check the mapping coefficients
        # first we need a process function that takes a frame and the m3 and m2 coefficients
        # and returns a B-scan; we can copose this out of several blob functions:
        def process_for_ui(frame,m3,m2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),[m3,m2,0.0,0.0]),dispersion_coefficients),0.9),oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)
        
        points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process_for_ui,c3min=-1e-9,c3max=1e-9,c2min=-1e-5,c2max=1e-5)

        # m2,m3 = points[np.argmax(maxes)]
        m2,m3 = points[-1]
        
        print('Optimized mapping coefficients:')
        print([m3,m2,0.0,0.0])
        sys.exit()

    if show_processed_data:
        processing_fig = plt.figure(0,figsize=(4,6))

    for frame_index in range(n_slow):
        print(frame_index)
        frame = src.get_frame(frame_index,diagnostics=diagnostics)
        frame = blob.dc_subtract(frame,diagnostics=diagnostics)
        frame = blob.k_resample(frame,mapping_coefficients,diagnostics=diagnostics)
        frame = blob.dispersion_compensate(frame,dispersion_coefficients,diagnostics=diagnostics)
        frame = blob.gaussian_window(frame,0.9,diagnostics=diagnostics)

        
        bscan_series = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)

        if not do_angiography:
            bscan = bscan_series
            bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)

        else:
        
            stack_complex = blob.reshape_repeats(bscan_series,n_repeats,x1=bscan_x1,x2=bscan_x2)


            if do_rigid_registration:
                ref = stack_complex[:,:,0]
                for k in range(1,stack_complex.shape[2]):
                    tar = stack_complex[:,:,k]
                    stack_complex[:,:,k] = rigid_shift(ref,tar,max_shift=10,diagnostics=False)

            bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_stack_%05d.npy'%frame_index)
            bscan = np.mean(np.abs(stack_complex),2)
            phase_variance = blob.make_angiogram(stack_complex,
                                                 bulk_correction_threshold=bulk_correction_threshold,
                                                 phase_variance_threshold=phase_variance_threshold,
                                                 diagnostics=diagnostics)

            angiogram_out_filename = os.path.join(output_directory_angiograms,'angiogram_bscan_%05d.npy'%frame_index)


        if show_processed_data:
            if do_angiography:
                png_out_filename = os.path.join(output_directory_png,'bscan_angiogram_%05d.png'%frame_index)
            else:
                png_out_filename = os.path.join(output_directory_png,'bscan_%05d.png'%frame_index)
            
            plt.figure(0)

            plt.clf()
            if do_angiography:
                plt.subplot(2,1,1)
            plt.imshow(20*np.log10(np.abs(bscan)),aspect='auto',cmap='gray',clim=(40,90))
            plt.colorbar()
            plt.title('bscan dB')
            plt.xticks([])
            plt.yticks([])

            if do_angiography:
                plt.subplot(2,1,2)
                plt.imshow(phase_variance,aspect='auto',cmap='gray')
                plt.colorbar()
                plt.title('angiogram (pv)')
                plt.xticks([])
                plt.yticks([])

            plt.savefig(png_out_filename,dpi=150)
            
            plt.pause(.1)

        if do_angiography:
            # here we're saving the complex stack--could abs and average them first if we need to save disk space
            np.save(bscan_out_filename,stack_complex)
        else:
            np.save(bscan_out_filename,bscan)
            
        if do_angiography:
            np.save(angiogram_out_filename,phase_variance)

        if diagnostics:
            plt.show()



def proc(fn):
    return process(fn,diagnostics=False,manual_dispersion=False,manual_mapping=False,n_skip=0)

flist = sorted(glob.glob(file_search_string))

# to do diagnostics, do something like the following:
# process(flist[0],diagnostics=True)
# sys.exit()

# to do manual dispersion compensation, do something like the following:
# process(flist[0],diagnostics=False,manual_dispersion=True)
# sys.exit()

# to do manual mapping, do something like the following:
process(flist[0],diagnostics=False,manual_mapping=True)
sys.exit()

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
        process(unp_filename,diagnostics=False,manual_dispersion=False,n_skip=0)
