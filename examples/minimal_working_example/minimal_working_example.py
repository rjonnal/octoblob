# Processing file 16_53_25.unp acquired using conventional Axsun OCT system
# Minimal working example for this system

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


###################################################################
###################################################################
# Processing parameters
# This section contains all of the constants used to process this
# dataset. Some of these are derived from the XML file created during
# acquisition. The XML file is shown in this section as well.

# You have to modify the path below to point at your data:
filename = '/home/rjonnal/Dropbox/Data/conventional_org/flash/minimal_working_example/16_53_25.unp'

# The data (.unp file, and .xml file if desired) can be downloaded from:
# https://www.dropbox.com/scl/fo/o9nskz1bkw0mkfc6iqhir/h?rlkey=ijdhh1ta648ajlmvvqql3qu48&dl=0

# Data dimensions are recorded in separate 16_53_25.xml:
###### XML ######################################
# <?xml version="1.0" encoding="utf-8"?>
# <MonsterList>
#  <!--Program Generated Easy Monster-->
#  <Monster>
#   <Name>Goblin</Name>
#   <Time
#    Data_Acquired_at="9/21/2021 4:53:25 PM" />
#   <Volume_Size
#    Width="1536"
#    Height="250"
#    Number_of_Frames="400"
#    Number_of_Volumes="1"
#    BscanWidth="736"
#    BscanOffset="32" />
#   <Scanning_Parameters
#    X_Scan_Range="1907"
#    X_Scan_Offset="650"
#    Y_Scan_Range="0"
#    Y_Scan_Offset="-500"
#    Number_of_BM_scans="1" />
#   <Dispersion_Parameters
#    C2="-9E-06"
#    C3="3E-09" />
#   <Fixation_Target
#    X="32"
#    Y="64" />
#  </Monster>
# </MonsterList>
########## End XML #################################

# The parameters we take from the XML file are:
# n_vol (Number_of_Volumes) = 1
# n_slow (Number_of_Frames) = 400
# n_repeats (Number_of_BM_scans) = 1
# n_fast (Height) = 250
# n_depth (Width) = 1536

# We also have the following a priori information:
# The data type is unsigned integer (16 bit)
# Each 16-bit integer must be right-shifted 4 bits to express the digitized value;
# in other words, the 12 meaningful bits are put into the first 12 places in the
# 16-bit integer, effectively multiplying each pixel by 16.

n_vol = 1
n_slow = 400
n_repeats = 1
n_fast = 250
n_depth = 1536
dtype = np.uint16
bit_shift_right = 4
bytes_per_pixel = 2

# Describing the index of the frame we want, in terms of volume_index
# and frame_index: each UNP file may contain multiple volumes, so to get
# a single frame we need to index both the volume and the frame within
# that volume
volume_index = 0 # this file contains only one volume, so anything >0 causes error
frame_index = 50 # arbitrary frame between 0 and n_slow-1 (399, in this case)

# Where to crop the spectra before dispersion compensation, processing
# into B-scans, etc.
k_crop_1 = 100
k_crop_2 = 1490


# For FBG alignment, specify the maximum index (in the k dimension) where the FBG
# could be found and the correlation threshold required to assume two spectra,
# cropped at that index (i.e., containing only the FBG portion and not the main
# sample-induced fringes), are aligned with one another (i.e., requiring no shifting)
fbg_max_index = 150
fbg_region_correlation_threshold = 0.9

# Define (and create, if necessary, a folder for figures)
fig_folder = 'figures'
os.makedirs(fig_folder,exist_ok=True)

dB_clims = (40,None)

# End of processing parameters section
###################################################################
###################################################################


# Getting a single frame of raw data from the UNP file
# The UNP file has no header information, only the spectral data


# Calculate the entry point into the file:
bytes_per_volume = n_depth * n_fast * n_slow * bytes_per_pixel
bytes_per_frame = n_depth * n_fast * bytes_per_pixel
pixels_per_frame = n_depth * n_fast
position = volume_index * bytes_per_volume + frame_index * bytes_per_frame

# Open the file in a `with` block, using numpy's convenient binary-reading
# function `fromfile`:
with open(filename,'rb') as fid:
    fid.seek(position,0)
    frame = np.fromfile(fid,dtype=dtype,count=pixels_per_frame)

frame = np.right_shift(frame,bit_shift_right)

# Reshape the frame into the correct dimensions, transpose so that the k/lambda dimension is
# vertical, and cast as floating point type to avoid truncation errors in downstream calculations:
frame = np.reshape(frame,(n_fast,n_depth)).T
frame = frame.astype(float)


# A general note about figures in this plot. They'll all be in `if` blocks, so they can be
# turned on and off easily. Also, we create a new figure in each block, and save the call
# to `plt.show()` until the end of the script.

# If desired, show the frame and plot its average over x (i.e., average spectrum)
show_figures = True
if show_figures:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(frame,aspect='auto')
    plt.subplot(1,2,2)
    plt.plot(np.mean(frame,axis=1))
    plt.title('lateral mean')
    plt.suptitle('Raw data')
    plt.savefig(os.path.join(fig_folder,'raw_data.png'))
    
# The next step is to align the spectra to their FBG features. The spectra are cropped
# around the FBG feature (up to index 150 in the spectral scan), and the individual
# spectral scans are cross-correlated. This could be done by cross-correlating each
# one to a reference scan (e.g., the first one). In practice, it's faster to group them
# by correlation and cross-correlate the groups. It's a little more complicated than
# necessary, but speeds things up.
# Set a limit on the maximum index where the FBG trough could possibly be located.
# This is a critical parameter, as it avoids cross correlation of spectra based on
# structural information; this would prevent the FBG features from dominating the
# cross-correlation and introduce additional phase noise.
# Correlation threshold is the minimum correlation required to consider two spectra
# to be in phase with one another
# We'll package the FBG alignment into a function to keep things somewhat neat:
def fbg_align(spectra,max_index,correlation_threshold):
    # crop the frame to the FBG region
    f = spectra[:fbg_max_index,:].copy()

    # group the spectra by amount of shift
    # this step avoids having to perform cross-correlation operations on every
    # spectrum; first, we group them by correlation with one another
    # make a list of spectra to group
    to_do = list(range(f.shape[1]))
    # make a list for the groups of similarly shifted spectra
    groups = []
    ref = 0

    # while there are spectra left to group, do the following loop:
    while(True):
        groups.append([ref])
        to_do.remove(ref)
        for tar in to_do:
            c = np.corrcoef(f[:,ref],f[:,tar])[0,1]
            if c>correlation_threshold:
                groups[-1].append(tar)
                to_do.remove(tar)
        if len(to_do)==0:
            break
        ref = to_do[0]

    subframes = []
    for g in groups:
        subf = f[:,g]
        subframes.append(subf)

    # now decide how to shift the groups of spectra by cross-correlating their means
    # we'll use the first group as the reference group:
    group_shifts = [0]
    ref = np.mean(subframes[0],axis=1)
    # now, iterate through the other groups, compute their means, and cross-correlate
    # with the reference. keep track of the cross-correlation peaks in the list group_shifts
    for taridx in range(1,len(subframes)):
        tar = np.mean(subframes[taridx],axis=1)
        xc = np.fft.ifft(np.fft.fft(ref)*np.fft.fft(tar).conj())
        shift = np.argmax(xc)
        if shift>len(xc)//2:
            shift = shift-len(xc)
        group_shifts.append(shift)

    # now, use the groups and the group_shifts to shift all of the spectra according to their
    # group membership:
    for g,s in zip(groups,group_shifts):
        for idx in g:
            spectra[:,idx] = np.roll(spectra[:,idx],s)
            f[:,idx] = np.roll(f[:,idx],s)

    return spectra


# Use our function to align the spectra:
spectra = fbg_align(frame,max_index=fbg_max_index,correlation_threshold=fbg_region_correlation_threshold)

# show the FBG-aligned frame:
if show_figures:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(spectra,aspect='auto')
    plt.subplot(1,2,2)
    plt.plot(np.mean(spectra,axis=1))
    plt.title('lateral mean')
    plt.suptitle('FBG-aligned')
    plt.savefig(os.path.join(fig_folder,'fbg_aligned.png'))


# Now we DC-subtract the spectra. We estimate the DC by averaging the spectra together,
# and subtract it from each one (using [array broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
dc = spectra.mean(1)

spectra = (spectra.T-dc).T

# non-broadcasting version, for reference, which
# does the same thing as the broadcasting version above,
# but obviates the for-loop:
# for x in range(spectra.shape[1]):
#    spectra[:,x] = spectra[:,x]-dc



# show the DC-subtracted frame:
if show_figures:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(spectra,aspect='auto')
    plt.subplot(1,2,2)
    plt.plot(spectra[:,100])
    plt.title('scan 100')
    plt.suptitle('DC subtracted')
    plt.savefig(os.path.join(fig_folder,'dc_subtracted.png'))



# The next steps are optimization of mapping and dispersion coefficients. This will be
# done using numerical optimization. But in order to do that we need to write a function
# that takes our FBG-aligned/DC-subtracted spectra, mapping coefficients, and dispersion
# coefficients, and produces a B-scan. We need this function first because the objective
# function for optimization operates on the sharpness of the resulting B-scan.

# Mapping correction (k_resample)
# By "mapping" we mean the process by which we infer the wave number (k) at which each
# of our spectral samples were measured. We cannot in general assume that k is a linear
# function of sample index. This is obviously true in cases where the spectrum is sampled
# uniformly with respect to lambda, since k=(2 pi)/lambda. In those cases, we minimally
# require interpolation into uniformly sampled k space. However, we shouldn't generally
# assume uniform sampling in lambda either, since swept-sources like the Broadsweeper
# and spectrometers may not behave linearly in time/space. Even sources with k-clocks,
# such as the Axsun swept source, may have mapping errors.
# To correct the mapping error we do the following:
# 1. Interpolate from lambda-space into k-space (not required for the Axsun source used
#    to acquire these data).
# 2. Let s(m+e(m)) be the acquired spectrum, with indexing error e(m). We determine a polynomial
#    e(m) = c3*m^3+c2*m^2, with coefficients c3 and c2, and then we interpolate from s(m+e(m))
#    to s(m+e(m)-e(m))=s(m).

# Dispersion correction (dispersion_compensate)
# This is a standard approach, described in multiple sources [add citations]. We define a unit
# amplitude phasor exp[j (mc3*k^3 + mc2*k^2)] with two coefficients mc3 and mc2, and multiply this
# by the acquired spectra.

def k_resample(spectra,coefficients):
    # If all coefficients are 0, return the spectra w/o further computation:
    if not any(coefficients):
        return spectra

    # the coefficients passed into this function are just the 3rd and 2nd order ones; we
    # add zeros so that we can use convenience functions like np.polyval that handle the
    # algebra; the input coefficients are [mc3,mc2], either a list or numpy array;
    # cast as a list to be on the safe side.
    coefficients = list(coefficients) + [0.0,0.0]

    # For historic, MATLAB-related reasons, the index m is defined between 1 and the spectral
    # length. This is a good opportunity to mention 
    # x_in specified on array index 1..N+1
    x_in = np.arange(1,spectra.shape[0]+1)

    # define an error polynomial, using the passed coefficients, and then
    # use this polynomial to define the error at each index 1..N+1
    error = np.polyval(coefficients,x_in)
    x_out = x_in + error

    # using the spectra measured at indices x_in, interpolate the spectra at indices x_out
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    interpolator = spi.interp1d(x_in,spectra,axis=0,kind='cubic',fill_value='extrapolate')
    interpolated = interpolator(x_out)
    return interpolated
    
# Next we need to dispersion compensate; for historical reasons the correction polynomial
# is defined on index x rather than k, but for physically meaningful numbers we should
# use k instead
def dispersion_compensate(spectra,coefficients):
    # If all coefficients are 0, return the spectra w/o further computation:
    if not any(coefficients):
        return spectra

    # the coefficients passed into this function are just the 3rd and 2nd order ones; we
    # add zeros so that we can use convenience functions like np.polyval that handle the
    # algebra; the input coefficients are [dc3,dc2], either a list or numpy array;
    # cast as a list to be on the safe side.
    coefs = list(coefficients) + [0.0,0.0]
    # define index x:
    x = np.arange(1,spectra.shape[0]+1)
    
    # define the phasor and multiply by spectra using broadcasting:
    dechirping_phasor = np.exp(-1j*np.polyval(coefs,x))
    dechirped = (spectra.T*dechirping_phasor).T
        
    return dechirped


# Now we can define our B-scan making function, which consists of:
# 1. k-resampling
# 2. dispersion compensation
# 3. windowing (optionally)
# 3. DFT
# We package the mapping and dispersion coefficients into a single list or array,
# in this order: 3rd order mapping coefficient, 2nd order mapping coefficient,
# 3rd order dispersion coefficient, 2nd order dispersion coefficient
def spectra_to_bscan(spectra,mapping_dispersion_coefficients):
    mapping_coefficients = mapping_dispersion_coefficients[:2]
    dispersion_coefficients = mapping_dispersion_coefficients[2:]

    spectra = k_resample(spectra,mapping_coefficients)
    spectra = dispersion_compensate(spectra,dispersion_coefficients)

    if True:
        # use a sigma (standard deviation) equal to 0.9 times the half-width
        # of the spectrum; this is arbitrary and was selected empirically, sort of
        sigma = 0.9
        window = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma**2))
        # multiply by broadcasting:
        spectra = (spectra.T*window).T

    bscan = np.fft.fft(spectra,axis=0)

    # remove one of the conjugate pairs--the top (inverted) one, by default
    bscan = bscan[bscan.shape[0]//2:,:]
    
    return bscan

# Let's make a B-scan with 0 for all mapping and dispersion coefficients, and show
# it.
bscan_uncorrected = spectra_to_bscan(spectra,[0.0,0.0,0.0,0.0])

if show_figures:
    plt.figure()
    plt.imshow(20*np.log10(np.abs(bscan_uncorrected)),cmap='gray',clim=dB_clims,aspect='auto')
    plt.colorbar()
    plt.title('B-scan w/o mapping or dispersion correction')
    plt.savefig(os.path.join(fig_folder,'bscan_uncorrected.png'))


# Now we are ready to run a four parameter optimization of the mapping and dispersion
# coefficients.

# First we need an objective function--one to be minimized. It will be based on image
# sharpness.
def sharpness(im):
    """Image sharpness"""
    return np.sum(im**2)/(np.sum(im)**2)

def objective_function(mapping_dispersion_coefficients,spectra):
    bscan = spectra_to_bscan(spectra,mapping_dispersion_coefficients)
    bscan = np.abs(bscan)
    bscan_sharpness = sharpness(bscan)
    print(1.0/bscan_sharpness)
    return 1.0/bscan_sharpness # remember this is a minimization algorithm

# initial guess for mapping_dispersion_coefficients
initial_guess = [0.0,0.0,0.0,0.0]

# run the optimizer
result = spo.minimize(objective_function,initial_guess,args=(spectra))

# get the optimized coefficients
coefs = result.x

bscan_corrected = spectra_to_bscan(spectra,coefs)

if show_figures:
    plt.figure()
    plt.imshow(20*np.log10(np.abs(bscan_corrected)),cmap='gray',clim=dB_clims,aspect='auto')
    plt.colorbar()
    plt.title('B-scan w/ mapping and dispersion correction')
    plt.savefig(os.path.join(fig_folder,'bscan_corrected.png'))

    
# If the script made any figures, show them now:
if plt.gcf().number > 0:
    plt.show()

