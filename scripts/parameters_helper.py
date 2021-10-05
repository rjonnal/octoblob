from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader
import parameters as params
from octoblob.plotting_functions import despine,setup_plots

setup_plots('paper')

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

    src_uncropped = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None,bit_shift_right=params.bit_shift_right,dtype=params.dtype)

    test_uncropped = src_uncropped.get_frame(0).astype(np.float)
    
    plt.figure()
    plt.imshow(test_uncropped,aspect='auto')
    plt.title('Uncropped spectra. Note approximate location (row) of FBG trough (or None), \ndesired rows of spectrum_start and spectrum_end.')

    try:
        src_cropped = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None,bit_shift_right=params.bit_shift_right,dtype=params.dtype,spectrum_start=params.spectrum_start,spectrum_end=params.spectrum_end)
        test_cropped = src_cropped.get_frame(0).astype(np.float)
    except AttributeError as ae:
        print(ae)
        print('spectrum_start and/or spectrum_end not set in parameters.py; please set them and re-run this script.')
        plt.show()
        sys.exit()

    
    plt.figure()
    plt.imshow(test_cropped,aspect='auto')
    plt.title('Cropped spectra. Used to generate B-scan in Fig. 3.')


    # load and average a sample of of spectra, so axial eye movements are evident:
    counter = 0.0
    for k in range(0,n_slow,10):
        test_cropped = src_cropped.get_frame(k).astype(np.float)
        test_cropped = (test_cropped.T-np.mean(test_cropped,axis=1)).T
        try:
            bscan = bscan + np.abs(np.fft.fft(test_cropped,axis=0,n=params.fft_oversampling_size))
        except:
            bscan = np.abs(np.fft.fft(test_cropped,axis=0,n=params.fft_oversampling_size))
        counter = counter + 1.0
            
    log_bscan = 20*np.log10(bscan/counter)
    try:
        clim = params.png_dB_clim
    except:
        clim = (40,80)
        
    plt.figure()
    plt.imshow(log_bscan,aspect='auto',cmap='gray',clim=clim)

    try:
        z1 = params.bscan_z1
        z2 = params.bscan_z2
        x1 = params.bscan_x1
        x2 = params.bscan_x2
        plt.plot([x1,x2,x2,x1,x1],[z1,z1,z2,z2,z1],'y-')
    except Exception as e:
        print(e)
    
    plt.colorbar()
    plt.title('Note region of interest (bscan_z1, bscan_z2, bscan_x1, bscan_x2).')
    plt.show()


def make_report(filename):
    report_filename = filename.replace('.unp','')+'_report.pdf'

    left = 0.1
    width = 0.3
    left2 = 0.5
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

    src_uncropped = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None,bit_shift_right=params.bit_shift_right,dtype=params.dtype)

    test_uncropped = src_uncropped.get_frame(0).astype(np.float)
    
    plt.figure(figsize=(8.5,11),dpi=50)
    text_ax = plt.axes([left,0.8,0.9,0.15])
    text_ax.imshow(np.ones((100,100)),clim=[0,1],aspect='auto',cmap='gray')
    text_ax.set_xticks([])
    text_ax.set_yticks([])
    despine()

    def write(x,y,text):
        text_ax.text(x,y,text,ha='left',va='top')
    
    write(0,0,'file: %s'%filename)
    write(20,0,'uncropped info:')
    write(20,5,src_uncropped.get_info(True))
    
    uc_ax = plt.axes([left,0.6,width,0.2])
    imh = uc_ax.imshow(test_uncropped,aspect='auto')
    plt.colorbar(imh)
    uc_ax.set_title('uncropped spectra')
    despine()

    try:
        src_cropped = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None,bit_shift_right=params.bit_shift_right,dtype=params.dtype,spectrum_start=params.spectrum_start,spectrum_end=params.spectrum_end)
        test_cropped = src_cropped.get_frame(0).astype(np.float)
        write(40,0,'cropped info:')
        write(40,5,src_cropped.get_info(True))

        c_ax = plt.axes([left2,0.6,width,0.2])
        imh = c_ax.imshow(test_cropped,aspect='auto')
        plt.colorbar(imh)
        c_ax.set_title('cropped spectra')
        despine()
    except AttributeError as ae:
        print(ae)
        print('spectrum_start and/or spectrum_end not set in parameters.py; please set them and re-run this script.')
        plt.show()
        sys.exit()

    try:
        clim = params.png_dB_clim
    except:
        clim = (40,80)
        
    # load and average a sample of of spectra, so axial eye movements are evident:
    counter = 0.0
    for k in range(0,n_slow,10):
        test_cropped = src_cropped.get_frame(k).astype(np.float)
        test_cropped = (test_cropped.T-np.mean(test_cropped,axis=1)).T
        try:
            bscan = bscan + np.abs(np.fft.fft(test_cropped,axis=0,n=params.fft_oversampling_size))
        except:
            bscan = np.abs(np.fft.fft(test_cropped,axis=0,n=params.fft_oversampling_size))
            sb_ax = plt.axes([left,0.35,width,0.2])
            log_sb = 20*np.log10(bscan)
            sb_ax.imshow(log_sb,cmap='gray',clim=clim,aspect='auto')
            sb_ax.set_title('single bscan')
            despine()
            sbp_ax = plt.axes([left,0.05,width,0.2])
            sbp_ax.plot(bscan.mean(axis=1))
            despine()
            sbp_ax.set_title('bscan profile')
            
        counter = counter + 1.0

    bscan = bscan/counter
    log_bscan = 20*np.log10(bscan)

    ab_ax = plt.axes([left2,0.35,width,0.2])
    ab_ax.imshow(log_bscan,cmap='gray',clim=clim,aspect='auto')
    ab_ax.set_title('average bscan')
    despine()
    sab_ax = plt.axes([left2,0.05,width,0.2])
    sab_ax.plot(bscan.mean(axis=1))
    sab_ax.set_title('average bscan profile')
    despine()
    plt.savefig(report_filename,dpi=300)
    
    
preliminary_visualizations(unp_filename)
make_report(unp_filename)
