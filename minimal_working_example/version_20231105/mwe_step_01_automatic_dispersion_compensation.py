# Processing file 16_53_25.unp acquired using conventional Axsun OCT system
# Minimal working example for this system.
# This is version 2, in which the OCT processing functions have been moved to a
# separate file to improve readability and encapsulation. Please see functions.py
# for details of how these functions work.

# Python 3.10.9
# Numpy 1.23.5
# Matplotlib version 3.7.0
# Scipy version 1.10.0


import numpy as np
from matplotlib import pyplot as plt
import os,sys
import scipy.optimize as spo
import scipy.interpolate as spi

# print library version information
import platform
import numpy
import scipy
import matplotlib
print('Python %s'%platform.python_version())
print('Numpy %s'%numpy.__version__)
print('Scipy %s'%scipy.__version__)
print('Matplotlib %s'%matplotlib.__version__)

import functions as blobf

try:
    filename = sys.argv[1]
except:
    print('Please supply the filename at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
    sys.exit()


###################################################################
###################################################################
# Processing parameters
# Mot of the processing and data file parameters are now stored in functions.py


# The data (.unp file, and .xml file if desired) can be downloaded from:
# https://www.dropbox.com/scl/fo/o9nskz1bkw0mkfc6iqhir/h?rlkey=ijdhh1ta648ajlmvvqql3qu48&dl=0

# Describing the index of the frame we want, in terms of volume_index
# and frame_index: each UNP file may contain multiple volumes, so to get
# a single frame we need to index both the volume and the frame within
# that volume
volume_index = 0 # this file contains only one volume, so anything >0 causes error
frame_index = 50 # arbitrary frame between 0 and n_slow-1 (399, in this case)

# Define (and create, if necessary, a folder for figures)
fig_folder = 'figures'
os.makedirs(fig_folder,exist_ok=True)

dB_clims = (40,90)

# Do you want to see and show processing steps as matplotlib figures?
show_figures = True

# End of processing parameters section
###################################################################
###################################################################


spectra = blobf.get_frame(filename,frame_index)

# If desired, show the spectra and plot its average over x (i.e., average spectrum)
if show_figures:
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(spectra,aspect='auto')
    plt.subplot(1,3,2)
    middle_index = spectra.shape[1]//2
    plt.plot(spectra[:,middle_index])
    plt.title('column %d'%middle_index)
    plt.subplot(1,3,3)
    plt.plot(np.mean(spectra,axis=1))
    plt.title('lateral mean')
    plt.suptitle('Raw data')
    plt.savefig(os.path.join(fig_folder,'raw_data.png'))


# Now we DC-subtract the spectra.
spectra = blobf.dc_subtract(spectra)

# show the DC-subtracted spectra:
if show_figures:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(spectra,aspect='auto')
    plt.subplot(1,2,2)
    plt.plot(spectra[:,100])
    plt.title('scan 100')
    plt.suptitle('DC subtracted')
    plt.savefig(os.path.join(fig_folder,'dc_subtracted.png'))


spectra = blobf.crop_spectra(spectra)
# show the DC-subtracted spectra:
if show_figures:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(spectra,aspect='auto')
    plt.subplot(1,2,2)
    plt.plot(spectra[:,100])
    plt.title('scan 100')
    plt.suptitle('Cropped')
    plt.savefig(os.path.join(fig_folder,'cropped.png'))


# Let's make a B-scan with 0 for all resampling and dispersion coefficients, and show
# it.
bscan_uncorrected = blobf.spectra_to_bscan(spectra,[0.0,0.0,0.0,0.0])

if show_figures:
    plt.figure()
    plt.imshow(20*np.log10(np.abs(bscan_uncorrected)),cmap='gray',clim=dB_clims,aspect='auto')
    plt.colorbar()
    plt.title('B-scan w/o resampling or dispersion correction')
    plt.savefig(os.path.join(fig_folder,'bscan_uncorrected.png'))

# initial guess for resampling_dispersion_coefficients
initial_guess = [0.0,0.0,0.0,0.0]

coefs = blobf.optimize_resampling_dispersion_coefficients(spectra,initial_guess=initial_guess)

print('Optimized coefs: %s'%coefs)

bscan_corrected = blobf.spectra_to_bscan(spectra,coefs)

if show_figures:
    plt.figure()
    plt.imshow(20*np.log10(np.abs(bscan_corrected)),cmap='gray',clim=dB_clims,aspect='auto')
    plt.colorbar()
    plt.title('B-scan w/ resampling and dispersion correction')
    plt.savefig(os.path.join(fig_folder,'bscan_corrected.png'))

    
# If the script made any figures, show them now:
if plt.gcf().number > 0:
    plt.show()

print('Saving %s to mapping_dispersion_coefficients.txt'%coefs)
np.savetxt('mapping_dispersion_coefficients.txt',coefs)
