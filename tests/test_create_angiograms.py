from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader,dispersion_ui
from octoblob.bmp_tools import savebmp

# PARAMETERS FOR RAW DATA SOURCE
filename = './octa_test_set.unp'

cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

output_directory_bscans = filename.replace('.unp','')+'_bscan_stacks'
output_directory_angiograms = filename.replace('.unp','')+'_angiograms'

os.makedirs(output_directory_bscans,exist_ok=True)
os.makedirs(output_directory_angiograms,exist_ok=True)

n_vol = cfg['n_vol']
n_slow = cfg['n_slow']
n_repeats = cfg['n_bm_scans']
n_fast = cfg['n_fast']
n_depth = cfg['n_depth']

# some conversions to comply with old conventions:
n_slow = n_slow
n_fast = n_fast*n_repeats

n_skip = 500
bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

# PROCESSING PARAMETERS
mapping_coefficients = [12.5e-10,-12.5e-7,0.0,0.0]
#dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]
dispersion_coefficients = [-1.1688311688311674e-09, 7.862903225806458e-06, 0.0, 0.0]

if False:
    # check the dispersion coefficients
    # first we need a process function that takes a frame and the c3 and c2 coefficients
    # and returns a B-scan; we can copose this out of several blob functions:
    def process(frame,c3,c2):
        return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),mapping_coefficients),[c3,c2,0.0,0.0]),0.9))[800:1200,:]
    points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process)

    c2,c3 = points[np.argmax(maxes)]
    print('Optimized coefficients:')
    print([c3,c2,0.0,0.0])
    sys.exit()



fft_oversampling_size = 4096
bscan_z1 = 2900
bscan_z2 = -40
bscan_x1 = 0
bscan_x2 = -100

# parameters for bulk motion correction and phase variance calculation:
bulk_correction_threshold = 0.3
phase_variance_threshold = 0.43


# setting diagnostics to True will plot/show a bunch of extra information to help
# you understand why things don't look right

diagnostics = False

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

    bscan = np.mean(np.abs(stack_complex),2)
    
    phase_variance = blob.make_angiogram(stack_complex,
                                         bulk_correction_threshold=bulk_correction_threshold,
                                         phase_variance_threshold=phase_variance_threshold,
                                         diagnostics=diagnostics)
    
    bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_stack_%05d.npy'%frame_index)
    angiogram_out_filename = os.path.join(output_directory_angiograms,'angiogram_bscan_%05d.npy'%frame_index)


    
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(20*np.log10(np.abs(bscan)),aspect='auto',cmap='gray')
    plt.colorbar()
    plt.title('bscan dB')

    plt.subplot(2,1,2)
    plt.imshow(phase_variance,aspect='auto',cmap='gray')
    plt.colorbar()
    plt.title('angiogram (pv)')
    
    plt.pause(.1)

    # here we're saving the complex stack--could abs and average them first if we need to save disk space
    np.save(bscan_out_filename,stack_complex)
    
    np.save(angiogram_out_filename,phase_variance)
    
    if diagnostics:
        break
    
