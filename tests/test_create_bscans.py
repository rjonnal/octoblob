from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time,glob
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob

from octoblob import config_reader
from octoblob import dispersion_ui
from octoblob.bmp_tools import savebmp


# PARAMETERS FOR RAW DATA SOURCE
filename = 'oct_test_set.unp'


cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))
output_directory = filename.replace('.unp','')+'_bscans'

try:
    os.mkdir(output_directory)
except Exception as e:
    pass


n_vol = cfg['n_vol']
n_slow = cfg['n_slow']
n_repeats = cfg['n_bm_scans']
n_fast = cfg['n_fast']
n_depth = cfg['n_depth']
n_skip = 0

bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

# PROCESSING PARAMETERS
mapping_coefficients = [12.5e-10,-12.5e-7,0.0,0.0]
# The first set here were from Justin's code, which
# works on the old OCTA data:
# dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]
# The second set was generated on 12/01/2020, using data
# collected a few days previously, right after Ravi and Kari
# swapped the balanced detection phase:
dispersion_coefficients = [5.1e-09, -7.65e-05, 0.0, 0.0]

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
bscan_z1 = 3000
bscan_z2 = 3800

bscan_x1 = 0
bscan_x2 = 400

# In this section, we will load one set of repeats and arrange them in a 3D array
# to be bulk-motion corrected

for frame_index in range(n_slow):
    print(frame_index)
    frame = src.get_frame(frame_index)
    frame = blob.dc_subtract(frame)
    frame = blob.k_resample(frame,mapping_coefficients)
    frame = blob.dispersion_compensate(frame,dispersion_coefficients)
    frame = blob.gaussian_window(frame,0.9)
    bscan = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2)
    out_filename = os.path.join(output_directory,'complex_bscan_%05d.npy'%frame_index)
    np.save(out_filename,bscan)
    bmp_filename = os.path.join(output_directory,'bscan_%05d.png'%frame_index)
    bscan_db = 20*np.log10(np.abs(bscan))
    savebmp(bmp_filename,bscan_db,clim=(40,80))
