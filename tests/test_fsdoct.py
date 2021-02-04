from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
from octoblob import config_reader,dispersion_ui
from octoblob.bmp_tools import savebmp
from octoblob import registration_tools as rt

import glob

#L0,dL,c3,c2 = [8.00649178e-07,6.00000000e-11,1.40844018e-09,5.82829653e-07]
L0,dL,c3,c2 = [8.01935799e-07,8.00000000e-11,-2.58738949e-09,-1.61729114e-06]

# PARAMETERS FOR RAW DATA SOURCE
filename = '/home/rjonnal/Dropbox/stimulus_sdoct/16_09_38-stimulus_dense.unp'

cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

output_directory_bscans = filename.replace('.unp','')+'_bscans'
output_directory_bscans_png = filename.replace('.unp','')+'_bscans_png'

os.makedirs(output_directory_bscans,exist_ok=True)
os.makedirs(output_directory_bscans_png,exist_ok=True)

n_vol = cfg['n_vol']
n_slow = cfg['n_slow']
n_repeats = cfg['n_bm_scans']
n_fast = cfg['n_fast']
n_depth = cfg['n_depth']

# some conversions to comply with old conventions:
n_slow = n_slow
n_fast = n_fast*n_repeats

n_skip = 0
bit_shift_right = 0
dtype=np.uint16

fbg_position = None
spectrum_start = 0
spectrum_end = 2048

src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

## We need this for spectral domain data:
#resampler = blob.Resampler(8.01e-7,4.1e-11,n_depth)
resampler = blob.Resampler(L0,dL,n_depth)

#self.wavelength_spectrum = np.polyval([4.1e-11,8.01e-7],np.arange(points_per_spectrum))

# PROCESSING PARAMETERS
mapping_coefficients = [12.5e-10,-12.5e-7,0.0,0.0]
#dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]
#dispersion_coefficients = [-1.1688311688311674e-09, 7.862903225806458e-06, 0.0, 0.0]
#dispersion_coefficients = [4.466265441875208e-10, 9.93279569892473e-06, 0.0, 0.0]
#dispersion_coefficients = [5.648664232173748e-09, -5.5913978494623534e-06, 0.0, 0.0]
dispersion_coefficients = [c3,c2, 0.0, 0.0]

if False:
    frame_index = 200
    # check the dispersion coefficients
    # first we need a process function that takes a frame and the c3 and c2 coefficients
    # and returns a B-scan; we can copose this out of several blob functions:
    def process(frame,c3,c2):
        #return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),mapping_coefficients),[c3,c2,0.0,0.0]),0.9))
        return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(resampler.map(blob.dc_subtract(frame)),[c3,c2,0.0,0.0]),0.9))[1750:2000,:]
    points,maxes = dispersion_ui.dispersion_ui(src.get_frame(frame_index,diagnostics=True),process)

    c2,c3 = points[np.argmax(maxes)]
    print('Optimized coefficients:')
    print([c3,c2,0.0,0.0])
    sys.exit()

fft_oversampling_size = 2048
bscan_z1 = 1700
bscan_z2 = -100
bscan_x1 = 0
bscan_x2 = -100

# parameters for bulk motion correction and phase variance calculation:
bulk_correction_threshold = 0.3
phase_variance_threshold = 0.43


# setting diagnostics to True will plot/show a bunch of extra information to help
# you understand why things don't look right, and then quit after the first loop
diagnostics = False

# setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look
show_processed_data = True

if show_processed_data:
    processing_fig = plt.figure(0,figsize=(4,3))

# In this section, we will load one set of repeats and arrange them in a 3D array
# to be bulk-motion corrected

for frame_index in range(n_slow):
    print(frame_index)
    bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)
    if os.path.exists(bscan_out_filename):
        continue
    
    frame = src.get_frame(frame_index)
    frame = blob.dc_subtract(frame)
    #frame = blob.k_resample(frame,mapping_coefficients)
    frame = resampler.map(frame)
    frame = blob.dispersion_compensate(frame,dispersion_coefficients,diagnostics=diagnostics)
    frame = blob.gaussian_window(frame,0.9)
    bscan_complex = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2,diagnostics=diagnostics)

    bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)
    bscan_png_out_filename = os.path.join(output_directory_bscans_png,'bscan_%05d.png'%frame_index)

    if show_processed_data:
        plt.figure(0)
        
        plt.clf()
        plt.imshow(20*np.log10(np.abs(bscan_complex)),aspect='auto',cmap='gray',clim=(50,90))
        plt.colorbar()
        plt.title('bscan dB')
        plt.savefig(bscan_png_out_filename,dpi=150)
        plt.pause(.1)

    # here we're saving the complex stack--could abs and average them first if we need to save disk space
    np.save(bscan_out_filename,bscan_complex)
    
    if diagnostics:
        plt.show()
        break
    
# Now we do point-by-point registration between pairs of B-scans, including phase alignment

flist = sorted(glob.glob(os.path.join(output_directory_bscans,'*.npy')))
rt.register_series(flist[300],flist,max_shift=50,overwrite=True,diagnostics=False)



sys.exit()

for fn1,fn2 in zip(flist[:-1],flist[1:]):
    f1 = np.load(fn1)
    f2 = np.load(fn2)
    xshift,yshift = rt.rigid_register(f1,f2,max_shift=10,diagnostics=False)

    rt.point_register(f1,f2)
    
    
