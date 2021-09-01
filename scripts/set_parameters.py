from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader
import parameters as params

unp_filename = sys.argv[1]

# This script explores the blob OCTRawData frames to help set parameters in parameters.py

def preliminary_visualizations(filename):
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

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None,bit_shift_right=params.bit_shift_right,dtype=params.dtype)

    test = src.get_frame(0).astype(np.float)
    plt.figure()
    plt.imshow(test,aspect='auto')
    plt.title('Note approximate location (row) of FBG trough (or None), \ndesired rows of spectrum_start and spectrum_end.')

    test = (test.T-np.mean(test,axis=1)).T
    bscan = 20*np.log10(np.abs(np.fft.fft(test,axis=0,n=params.fft_oversampling_size)))
    plt.figure()
    plt.imshow(bscan,aspect='auto',cmap='gray',clim=(40,80))
    plt.colorbar()
    plt.title('Note region of interest (bscan_z1, bscan_z2, bscan_x1, bscan_x2).')
    plt.show()

preliminary_visualizations(unp_filename)
