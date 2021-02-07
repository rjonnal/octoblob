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

bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

fft_oversampling_size = 4096
bscan_z1 = 2900
bscan_z2 = -300
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

def process_unp(filename,diagnostics,show_processed_data=True,manual_dispersion=False,manual_mapping=False,n_skip=0):

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
        
        points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process,c3min=-1e-7,c3max=1e-7,c2min=-1e-4,c2max=1e-4)

        c2,c3 = points[np.argmax(maxes)]
        print('Optimized dispersion coefficients:')
        print([c3,c2,0.0,0.0])
        sys.exit()


    if manual_mapping:
        # check the mapping coefficients
        # first we need a process function that takes a frame and the m3 and m2 coefficients
        # and returns a B-scan; we can copose this out of several blob functions:
        def process(frame,m3,m2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),[m3,m2,0.0,0.0]),dispersion_coefficients),0.9))[800:1200,:]
        
        points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process,c3min=-1e-9,c3max=1e-9,c2min=-1e-6,c2max=1e-6)

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
        bscan = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)[:,bscan_x1:bscan_x2]

        bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)
        
        if show_processed_data:
            png_out_filename = os.path.join(output_directory_png,'bscan_%05d.png'%frame_index)
            
            plt.figure(0)

            plt.clf()
            plt.imshow(20*np.log10(np.abs(bscan)),aspect='auto',cmap='gray',clim=(40,80))
            plt.colorbar()
            plt.title('bscan dB')
            plt.xticks([])
            plt.yticks([])
            
            plt.savefig(png_out_filename,dpi=150)
            
            plt.pause(.1)

        # here we're saving the complex stack--could abs and average them first if we need to save disk space
        np.save(bscan_out_filename,bscan)

        if diagnostics:
            plt.show()




def proc(fn):
    return process_unp(fn,diagnostics=False,manual_dispersion=False,n_skip=0)

flist = sorted(glob.glob('structural/*.unp'))

# to do diagnostics, do something like the following:
#process_unp(flist[0],diagnostics=True)
#sys.exit()

# to do manual dispersion compensation, do something like the following:
# process_unp(flist[0],diagnostics=False,manual_dispersion=True)
# sys.exit()

# to do manual mapping, do something like the following:
# process_unp(flist[0],diagnostics=False,manual_mapping=True)
# sys.exit()



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
        process_unp(unp_filename,diagnostics=False,manual_dispersion=False,n_skip=0)
