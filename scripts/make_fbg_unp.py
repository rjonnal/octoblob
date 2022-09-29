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
import parameters as params
from octoblob.bmp_tools import add_colorbar
from octoblob import plotting_functions as opf
import os,shutil


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


    output_filename = filename.replace('.unp','')+'_fbg.unp'
    xml_filename = filename.replace('.unp','.xml')
    
    assert os.path.exists(xml_filename)
    
    xml_output_filename = xml_filename.replace('.xml','')+'_fbg.xml'
    
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

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params.fbg_position,spectrum_start=None,spectrum_end=None,bit_shift_right=0,n_skip=params.n_skip,dtype=params.dtype)

    frames = []
    
    for frame_index in range(n_slow):
        if frame_index==end_frame:
            break
        print(frame_index)
        frame = src.get_frame(frame_index,diagnostics=diagnostics).astype(params.dtype)
        frames.append(frame)

    frames = np.array(frames,dtype=params.dtype)

    frames = np.transpose(frames,(0,2,1)).ravel()

    shutil.copyfile(xml_filename,xml_output_filename)
    
    frames.tofile(output_filename)
        
if __name__=='__main__':

    process(unp_filename)
