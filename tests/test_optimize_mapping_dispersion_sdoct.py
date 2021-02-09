from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader,dispersion_ui
from octoblob.bmp_tools import savebmp
import scipy.optimize as spo

# In this script we try to optimize four parameters of the SD-OCT processing pipeline:
# dispersion compensation coefficients c3 and c2
# spectral values L0 and dL

# some prelminaries to pull a set of spectra from the raw data source:

# PARAMETERS FOR RAW DATA SOURCE
filename = '/home/rjonnal/Dropbox/stimulus_sdoct/16_09_38-stimulus_dense.unp'

cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))
n_vol = cfg['n_vol']
n_slow = cfg['n_slow']
n_repeats = cfg['n_bm_scans']
n_fast = cfg['n_fast']
n_depth = cfg['n_depth']
n_skip = 0
bit_shift_right = 0
dtype=np.uint16

fbg_position = None
spectrum_start = 0
spectrum_end = 2048

src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

# define an objective function helper that takes a frame and the four optimization parameters
# and returns the figure of merit to minimize, in this case 1/max_intensity
def objective_helper(frame,L0,dL,c3,c2):
    resampler = blob.Resampler(L0,dL,frame.shape[0])
    frame = blob.dc_subtract(frame)
    frame = resampler.map(frame)
    frame = blob.dispersion_compensate(frame,[c3,c2,0,0])
    frame = blob.gaussian_window(frame,0.9)
    bscan = np.abs(blob.spectra_to_bscan(frame))[frame.shape[0]//2:-10,:]
    val = 1.0/bscan.max()
    print(L0,dL,c3,c2,val)
    if False:
        plt.clf()
        plt.imshow(bscan)
        plt.colorbar()
        plt.title([L0,dL,c3,c2])
        plt.pause(.1)
    return val

# pull an arbitrary frame
frame_index = 100
frame = src.get_frame(frame_index)

# curry the frame out of the signature
objective = lambda x: objective_helper(frame,x[0],x[1],x[2],x[3])

# define the limits for the four parameters
L0_lims = [790e-9,810e-9]
dL_lims = [2e-11,8e-11]
c3_lims = [-1e-8,1e-8]
c2_lims = [-1e-5,1e-5]

bounds=[L0_lims,dL_lims,c3_lims,c2_lims]
x0 = [(a[0]+a[1])/2.0 for a in bounds]

#result = spo.minimize(objective,x0,bounds=bounds,method='Newton-CG')
result = spo.dual_annealing(objective,bounds=bounds)
print(result.x)
sys.exit()

