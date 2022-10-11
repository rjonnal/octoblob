from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.optimize as spo
import sys,os
import pathlib

dB_lims = (45,80)
fbg_search_distance = 11
noise_samples = 80
output_folder = 'crawler_output'

unp_files = pathlib.Path('.').rglob('*.unp')

os.makedirs(output_folder,exist_ok=True)
for unp_file in unp_files:

    # make a folder for diagnostics
    unp_file = str(unp_file)
    diagnostics_folder = unp_file.replace('.unp','')+'_diagnostics'
    os.makedirs(diagnostics_folder,exist_ok=True)

    # get a source for raw frames
    src = blobf.get_source(unp_file)

    
    while src.has_more_frames():
        spectra = src.next_frame()
        print(src.current_frame_index)
        
    spectra = blobf.fbg_align(spectra,fbg_search_distance,noise_samples=noise_samples,diagnostics_path=path)
    
    
