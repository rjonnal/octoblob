# Processing file 16_53_25.unp acquired using conventional Axsun OCT system
# Minimal working example for this system

# Python 3.10.9
# Numpy 1.23.5
# Matplotlib version 3.7.0
import numpy as np
from matplotlib import pyplot as plt
import os,sys

###################################################################
###################################################################
# Processing parameters
# This section contains all of the constants used to process this
# dataset. Some of these are derived from the XML file created during
# acquisition. The XML file is shown in this section as well.

filename = '/home/rjonnal/Dropbox/Data/conventional_org/flash/minimal_working_example/16_53_25.unp'


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

# Mapping correction
# By "mapping" we mean the process by which we infer the wave number (k) at which each
# of our spectral samples were measured. We cannot in general assume that k is a linear
# function of sample index. This is obviously true in cases where the spectrum is sampled
# uniformly with respect to lambda, since k=(2 pi)/lambda. In those cases, we minimally
# require interpolation into uniformly sampled k space. However, we shouldn't generally
# assume uniform sampling in lambda either, since swept-sources like the Broadsweeper
# and spectrometers may not behave linearly in time/space.


def spectra_to_bscan(spectra,

    
# If the script made any figures, show them now:
if plt.gcf().number > 0:
    plt.show()

