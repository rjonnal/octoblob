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
import parameters as params


unp_filename = sys.argv[1]


def process(filename):
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

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params.fbg_position,spectrum_start=params.spectrum_start,spectrum_end=params.spectrum_end,bit_shift_right=params.bit_shift_right,n_skip=params.n_skip,dtype=params.dtype)

    def process_for_mapping_ui(frame,m3,m2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),[m3,m2,0.0,0.0]),[0.0,0.0,0.0,0.0]),0.9),oversampled_size=params.fft_oversampling_size,z1=params.bscan_z1,z2=params.bscan_z2)
    points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process_for_mapping_ui,
                                               params.m3min,params.m3max,params.m2min,params.m2max,
                                               'Select mapping coefficients; results of final click will be printed.')

    winner = -1
    try:
        m2,m3 = points[winner][0],points[winner][1]
        print('Last clicked mapping coefficients:')
    except:
        m2,m3 = 0.0,0.0
        print('Default mapping coefficients:')
    print('mapping_coefficients = [%0.1e, %0.1e, 0.0, 0.0]'%(m3,m2))
    #mapping_coefficients = [m3,m2,0.0,0.0]
    mapping_coefficients = [0.0,0.0,0.0,0.0]

    def process_for_dispersion_ui(frame,c3,c2):
            return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),mapping_coefficients),[c3,c2,0.0,0.0]),0.9),oversampled_size=params.fft_oversampling_size,z1=params.bscan_z1,z2=params.bscan_z2)
    points,maxes = dispersion_ui.dispersion_ui(src.get_frame(0),process_for_dispersion_ui,
                                               params.c3min,params.c3max,params.c2min,params.c2max,
                                               'Select dispersion coefficients; results of final click will be printed.')


    winner = -1
    try:
        c2,c3 = points[winner][0],points[winner][1]
        print('Last clicked dispersion coefficients:')
    except:
        c2,c3 = 0.0,0.0
        print('Default dispersion coefficients:')
        
    print('dispersion_coefficients = [%0.1e, %0.1e, 0.0, 0.0]'%(c3,c2))
    dispersion_coefficients = [c3,c2,0.0,0.0]

process(unp_filename)
