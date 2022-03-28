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
from octoblob.bmp_tools import add_colorbar
from octoblob import plotting_functions as opf

opf.setup_plots('paper')
opf.print_dpi = 120
opf.IPSP = 3.5

unp_filename = sys.argv[1]

flags = [t.lower() for t in sys.argv[2:]]

end_frame = np.inf
for f in flags:
    try:
        end_frame = int(f)
    except:
        pass


diagnostics = 'diagnostics' in flags
show_processed_data = 'show' in flags

try:
    png_aspect_ratio = params.png_aspect_ratio
except:
    png_aspect_ratio = 1.0

try:
    png_dB_clim = params.png_dB_clim
except:
    png_dB_clim = (40,90)


    
def process(filename,diagnostics=diagnostics,show_processed_data=show_processed_data):
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

    output_directory_bscans = filename.replace('.unp','')+'_bscans'
    os.makedirs(output_directory_bscans,exist_ok=True)

    output_directory_info = filename.replace('.unp','')+'_info'
    os.makedirs(output_directory_info,exist_ok=True)

    if show_processed_data:
        output_directory_png = filename.replace('.unp','')+'_png'
        os.makedirs(output_directory_png,exist_ok=True)

    diagnostics_base = diagnostics
    if diagnostics_base:
        diagnostics_directory = filename.replace('.unp','')+'_diagnostics'
        os.makedirs(diagnostics_directory,exist_ok=True)

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params.fbg_position,fbg_region_height=params.fbg_region_height,spectrum_start=params.spectrum_start,spectrum_end=params.spectrum_end,bit_shift_right=params.bit_shift_right,n_skip=params.n_skip,dtype=params.dtype)

    if show_processed_data:
        processing_fig = plt.figure(0)

    for frame_index in range(n_slow):
        if frame_index==end_frame:
            break
        if diagnostics_base:
            diagnostics = (diagnostics_directory,frame_index)
        print(frame_index)
        frame = src.get_frame(frame_index,diagnostics=diagnostics)
        frame = blob.dc_subtract(frame,diagnostics=diagnostics)
        frame = blob.k_resample(frame,params.mapping_coefficients,diagnostics=diagnostics)
        frame = blob.dispersion_compensate(frame,params.dispersion_coefficients,diagnostics=diagnostics)
        frame = blob.gaussian_window(frame,0.9,diagnostics=diagnostics)
        bscan = blob.spectra_to_bscan(frame,oversampled_size=params.fft_oversampling_size,z1=params.bscan_z1,z2=params.bscan_z2,diagnostics=diagnostics)
        bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)
        np.save(bscan_out_filename,bscan)
        
        if show_processed_data:
            png_out_filename = os.path.join(output_directory_png,'bscan_%05d.png'%frame_index)
            
            plt.figure(0)
            processing_fig.clear()
            im = plt.imshow(20*np.log10(np.abs(bscan)),aspect=png_aspect_ratio,cmap='gray',clim=png_dB_clim)
            plt.colorbar(im,fraction=0.03)
            plt.title('bscan dB')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(png_out_filename,dpi=150)
            np.save(bscan_out_filename,bscan)
            
        if diagnostics_base:
            # use plt.close('all') instead of plt.show() if you want to save the diagnostic plots
            # without seeing them
            plt.close('all')
            #plt.show()

        if show_processed_data:
            plt.pause(.001)

if __name__=='__main__':

    process(unp_filename)
