from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader,dispersion_tools
from octoblob.bmp_tools import savebmp
import scipy.optimize as spo

# In this script we try to optimize four parameters of the SD-OCT processing pipeline:
# dispersion compensation coefficients c3 and c2
# spectral values L0 and dL

# some prelminaries to pull a set of spectra from the raw data source:

# PARAMETERS FOR RAW DATA SOURCE
filename = './15_30_14-.unp'

cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))
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

# define an objective function helper that takes a frame and the four optimization parameters
# and returns the figure of merit to minimize, in this case 1/max_intensity
def objective_helper(frame,m3,m2,c3,c2):
    frame = blob.dc_subtract(frame)
    frame = blob.k_resample(frame,[m3,m2,0,0])
    frame = blob.dispersion_compensate(frame,[c3,c2,0,0])
    frame = blob.gaussian_window(frame,0.9)
    bscan = np.abs(blob.spectra_to_bscan(frame))[frame.shape[0]//2:-10,:]
    val = 1.0/bscan.max()
    print(m3,m2,c3,c2,val)
    if False:
        plt.clf()
        plt.imshow(bscan)
        plt.colorbar()
        plt.title([m3,m2,c3,c2])
        plt.pause(.1)
    return val

# pull an arbitrary frame
frame_index = 100
frame = src.get_frame(frame_index)

# curry the frame out of the signature
objective = lambda x: objective_helper(frame,x[0],x[1],x[2],x[3])

# define the limits for the four parameters
mapping_coefficients = [12.5e-10,-12.5e-7,0.0,0.0]
m3_lims = [-1e-9,1e-9]
m2_lims = [-1e-6,1e-6]
c3_lims = [-1e-8,1e-8]
c2_lims = [-1e-5,1e-5]

bounds=[m3_lims,m2_lims,c3_lims,c2_lims]
x0 = [(a[0]+a[1])/2.0 for a in bounds]

result = spo.dual_annealing(objective,bounds=bounds)
print(result.x)
sys.exit()

