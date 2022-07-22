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
import os,shutil

use_multiprocessing = params.use_multiprocessing
if use_multiprocessing:
    import multiprocessing as mp

opf.setup_plots('paper')
opf.print_dpi = 120
opf.IPSP = 3.5


args = sys.argv[1:]
args = blob.expand_wildcard_arguments(args)

files = []
flags = []

for arg in args:
    if os.path.exists(arg):
        files.append(arg)
    else:
        flags.append(arg.lower())
        

diagnostics = 'diagnostics' in flags
show_processed_data = 'show' in flags


end_frame = np.inf
for f in flags:
    try:
        end_frame = int(f)
    except:
        pass


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
        if diagnostics:
            plt.show()

    frames = np.array(frames,dtype=params.dtype)

    frames = np.transpose(frames,(0,2,1)).ravel()

    shutil.copyfile(xml_filename,xml_output_filename)
    
    frames.tofile(output_filename)

files.sort()

try:
    n_workers = params.multiprocessing_n_processes
except Exception as e:
    n_workers = 4

def proc(f):
    process(f,diagnostics=diagnostics,show_processed_data=show_processed_data)

    
if __name__=='__main__':
    
    if use_multiprocessing:
        with mp.Pool(n_workers) as p:
            p.map(proc,files)

    else:
        for f in files:
            process(f,diagnostics=diagnostics,show_processed_data=show_processed_data)
