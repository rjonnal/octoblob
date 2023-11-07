# This file contains functions to be used in the minimal working example for this system.

# Python 3.11.4
# Numpy 1.24.3
# Scipy 1.10.1
# Matplotlib 3.7.1

import numpy as np
from matplotlib import pyplot as plt
import os,sys
import scipy.optimize as spo
import scipy.interpolate as spi
from xml.etree import ElementTree as ET
import inspect

# print library version information
import platform
import numpy
import scipy
import matplotlib
print('Python %s'%platform.python_version())
print('Numpy %s'%numpy.__version__)
print('Scipy %s'%scipy.__version__)
print('Matplotlib %s'%matplotlib.__version__)


##### Start data file parameters ######

# These parameters will be used to extract frames from the raw data file.

dtype = np.uint16
bit_shift_right = 4
# bit shifting means moving the 12 significant bits from positions 0-11 in the 16 bit integer
# to positions 4-15, e.g. the number 4095 is 111111111111 in binary. This can be represented
# in 16 bits as 0000111111111111 or 1111111111110000. Alazar represents it the second way, which
# causes the digitized values to be 16x larger than they should be. To correct this, we shift
# bits 4 places to the right: 1111111111110000 -> 0000111111111111
bytes_per_pixel = 2

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

##### End data file parameters ######



def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0


def get_configuration(filename):

    ''' Pull configuration parameters from Yifan's
    config file. An example configuration file is shown
    below. Calling get_configuration('temp.xml') returns
    a dictionary of parameters useful for processing the OCT
    stack, e.g. numbers of scans in x and y directions,
    voltage range of scanners, etc.

    Example XML config file:

    <?xml version="1.0" encoding="utf-8"?>
    <MonsterList>
     <!--Program Generated Easy Monster-->
     <Monster>
      <Name>Goblin</Name>
      <Time
       Data_Acquired_at="1/30/2018 12:21:22 PM" />
      <Volume_Size
       Width="2048"
       Height="400"
       Number_of_Frames="800"
       Number_of_Volumes="1" />
      <Scanning_Parameters
       X_Scan_Range="3000"
       X_Scan_Offset="0"
       Y_Scan_Range="0"
       Y_Scan_Offset="0"
       Number_of_BM_scans="2" />
      <Dispersion_Parameters
       C2="0"
       C3="0" />
     </Monster>
    </MonsterList>

    Example output dictionary:

    {'y_offset_mv': 0, 'x_offset_mv': 0, 'n_fast': 400, 
     'y_scan_mv': 0, 'n_slow': 800, 'n_vol': 1, 
     'x_scan_mv': 3000, 'time_stamp': '1/30/2018 12:21:22 PM', 
     'n_bm_scans': 2, 'n_depth': 2048}

    '''
    
    XML_DICT = {}
    # populate XML_DICT with required parameters from Yifan's XML grammar
    # keys of this dictionary [x,y] are x = element tag and y = element attribute
    # the values of this dictionary (x,y) are x = our new name for the data and
    # y = the data type (i.e. a function that we can cast the output with)
    XML_DICT['Time','Data_Acquired_at'] = ('time_stamp',str)
    XML_DICT['Volume_Size','Width'] = ('n_depth',int)
    XML_DICT['Volume_Size','Height'] = ('n_fast',int)
    XML_DICT['Volume_Size','Number_of_Frames'] = ('n_slow',int)
    XML_DICT['Volume_Size','Number_of_Volumes'] = ('n_vol',int)
    XML_DICT['Scanning_Parameters','X_Scan_Range'] = ('x_scan_mv',int)
    XML_DICT['Scanning_Parameters','X_Scan_Offset'] = ('x_offset_mv',int)
    XML_DICT['Scanning_Parameters','Y_Scan_Range'] = ('y_scan_mv',int)
    XML_DICT['Scanning_Parameters','Y_Scan_Offset'] = ('y_offset_mv',int)
    XML_DICT['Scanning_Parameters','Number_of_BM_scans'] = ('n_bm_scans',int)

    # append extension if it's not there
    if not filename[-4:].lower()=='.xml':
        filename = filename + '.xml'

    
    # use Python's ElementTree to get a navigable XML tree
    temp = ET.parse(filename).getroot()

    # start at the root, called 'Monster' for whatever reason:
    tree = temp.find('Monster')

    # make an empty output dictionary
    config_dict = {}

    # iterate through keys of specification (XML_DICT)
    # and find corresponding settings in the XML tree.
    # as they are found, insert them into config_dict with
    # some sensible but compact names, casting them as
    # necessary:
    for xml_key in XML_DICT.keys():
        node = tree.find(xml_key[0])
        config_value = node.attrib[xml_key[1]]
        xml_value = XML_DICT[xml_key]
        config_key = xml_value[0]
        config_cast = xml_value[1]
        config_dict[config_key] = config_cast(config_value)
        
    return config_dict


# Lets make a dictionary of frames, so that if the same frame is requested more
# than once, for example during dispersion compensation optimization, we can quickly
# supply it. The unique identifier for the frame (frame_key) will be the tuple
# (filename, frame_index, volume_index).
frame_dict = {}

def get_frame(filename,frame_index,volume_index=0,bit_shift_right=4):
    frame_key = (filename,frame_index,volume_index)
    # first, try to get it from the dictionary, in case it's been read and fbg_aligned
    # already:
    try:
        return frame_dict[frame_key]
    # if it's not found in the dictionary, read the data, reshape, fbg-align, and store
    # it in the dictionary:
    except KeyError:
        cfg = get_configuration(filename.replace('.unp','.xml'))
        n_depth = cfg['n_depth']
        n_fast = cfg['n_fast']
        n_slow = cfg['n_slow']

        # Calculate the entry point into the file:
        pixels_per_frame = n_depth * n_fast
        pixels_per_volume = pixels_per_frame * n_slow
        bytes_per_volume = pixels_per_volume * bytes_per_pixel
        bytes_per_frame = pixels_per_frame * bytes_per_pixel
        position = volume_index * bytes_per_volume + frame_index * bytes_per_frame

        # Open the file in a `with` block, using numpy's convenient binary-reading
        # function `fromfile`:
        with open(filename,'rb') as fid:
            fid.seek(position,0)
            frame = np.fromfile(fid,dtype=dtype,count=pixels_per_frame)

        # frame is now a 1D Numpy array containing pixels_per_frame number of integer values

        frame = np.right_shift(frame,bit_shift_right)

        # Reshape the frame into the correct dimensions, transpose so that the k/lambda dimension is
        # vertical, and cast as floating point type to avoid truncation errors in downstream calculations:
        frame = np.reshape(frame,(n_fast,n_depth)).T

        # frame is now a 2D Numpy array containing n_depth (vertical) x n_fast (horizontal) integer values

        frame = frame.astype(float)
        # frame is now floating point
        frame = fbg_align(frame)
        frame_dict[frame_key] = frame
        
    return frame


def crop_spectra(frame):
    frame = frame[k_crop_1:k_crop_2,:]
    return frame
    

def fbg_align(spectra):
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
            if c>fbg_region_correlation_threshold:
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


def dc_subtract(spectra):

    # Now we DC-subtract the spectra. We estimate the DC by averaging the spectra together,
    # and subtract it from each one (using [array broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)).
    dc = spectra.mean(1)

    # A brief explanation of array broadcasting:
    # spectra is a 2D array with shape (n_depth,n_fast) and dc is a 1D array with shape (n_depth)
    # We want to subtract dc from each column of spectra. We can do this using array broadcasting.
    # Broadcasting allows us to perform element-wise operations on arrays of different shapes.
    # We can broadcast A-B if A has shape (x,y) and B has shape (y). It's not relevant here, but
    # the same idea applies to higher dimensionalities, e.g., if A has shape (x,y,z) and B has
    # shape (y,z) or (z).
    # In order to subtract dc from spectra we have to transpose spectra to shape (n_fast,n_depth), then
    # subtract, and then, transpose back to shape (n_depth,n_fast).
    spectra = (spectra.T-dc).T
    return spectra

# The next steps are optimization of resampling and dispersion coefficients. This will be
# done using numerical optimization. But in order to do that we need to write a function
# that takes our FBG-aligned/DC-subtracted spectra, resampling coefficients, and dispersion
# coefficients, and produces a B-scan. We need this function first because the objective
# function for optimization operates on the sharpness of the resulting B-scan.

# Resampling correction (k_resample)
# By "resampling" we mean the process by which we infer the wave number (k) at which each
# of our spectral samples were measured. We cannot in general assume that k is a linear
# function of sample index. This is obviously true in cases where the spectrum is sampled
# uniformly with respect to lambda, since k=(2 pi)/lambda. In those cases, we minimally
# require interpolation into uniformly sampled k space. However, we shouldn't generally
# assume uniform sampling in lambda either, since swept-sources like the Broadsweeper
# and spectrometers may not behave linearly in time/space. Even sources with k-clocks,
# such as the Axsun swept source, may have resampling errors.
# To correct the resampling error we do the following:
# 1. Interpolate from lambda-space into k-space (not required for the Axsun source used
#    to acquire these data).
# 2. Let s(m+e(m)) be the acquired spectrum, with indexing error e(m). We determine a polynomial
#    e(m) = c3*m^3+c2*m^2, with coefficients c3 and c2, and then we interpolate from s(m+e(m))
#    to s(m+e(m)-e(m))=s(m).

# Dispersion correction (dispersion_compensate)
# This is a standard approach, described in multiple sources [add citations]. We define a unit
# amplitude phasor exp[j (rc3*k^3 + rc2*k^2)] with two coefficients rc3 and rc2, and multiply this
# by the acquired spectra.

def k_resample(spectra,coefficients):
    # Coefficients is a 2-item list [rc3,rc2] containing the 3rd and 2nd order coefficients
    # of a 3rd order polynomial that specifies the index-k error. If both coefficients are 0,
    # then the error polynomial is all zero, and thus the input spectra should be returned.
    # If all coefficients are 0, return the spectra w/o further computation:
    if not any(coefficients):
        return spectra

    # the coefficients passed into this function are just the 3rd and 2nd order ones; we
    # add zeros so that we can use convenience functions like np.polyval that handle the
    # algebra; the input coefficients are [rc3,rc2], either a list or numpy array;
    # cast as a list to be on the safe side.
    coefficients = list(coefficients) + [0.0,0.0]

    # Now coefficients is a 4-item list, [rc3,rc2,0.0,0.0]

    # For historic, MATLAB-related reasons, the index m is defined between 1 and the spectral
    # length. This is a good opportunity to mention 
    # x_in specified on array index 1..N+1
    x_in = np.arange(1,spectra.shape[0]+1)

    # define an error polynomial, using the passed coefficients, and then
    # use this polynomial to define the error at each index 1..N+1
    error = np.polyval(coefficients,x_in)
    x_out = x_in + error

    # For example, our x_in is the array: 1, 2, 3, .... 1536
    # Our error polynomial may be the array: 1.8e-10,-3.5e-11,7.5e-10,.... 3.5e-10 
    # x_out is the sum of these two.

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

    # now coefs is a 4-item list: [dc3,dc2,0.0,0.0]
    
    # define index x:
    x = np.arange(1,spectra.shape[0]+1)

    # if we want to avoid using polyval, we can explicitly evaluate the polynomial:
    # evaluate our polynomial on index x; if we do it this way, we need not append
    # zeroes to coefs above
    # phase_polynomial = coefs[0]*x**3 + coefs[1]*x**2

    # actually it's simpler to use polyval, which is why we appended the zeros to
    # the input coefficients--polyval infers the order of the polynomial from the
    # number of values in the list/array:
    phase_polynomial = np.polyval(coefs,x)
    
    # define the phasor and multiply by spectra using broadcasting:
    dechirping_phasor = np.exp(-1j*phase_polynomial)
    dechirped = (spectra.T*dechirping_phasor).T
        
    return dechirped



def window_spectra(spectra,sigma=0.9):
    # use a sigma (standard deviation) equal to 0.9 times the half-width
    # of the spectrum; this is arbitrary and was selected empirically, sort of
    sigma = 0.9
    window = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma**2))
    # gaussian window defined on a set of numbers e.g. -1.0,-.99,-.98.....0.0... .98, .99, 1.0
    # multiply by broadcasting:
    spectra = (spectra.T*window).T
    return spectra


# Now we can define our B-scan making function, which consists of:
# 1. k-resampling
# 2. dispersion compensation
# 3. windowing (optionally)
# 3. DFT
# We package the resampling and dispersion coefficients into a single list or array,
# in this order: 3rd order resampling coefficient, 2nd order resampling coefficient,
# 3rd order dispersion coefficient, 2nd order dispersion coefficient
def spectra_to_bscan(spectra,resampling_dispersion_coefficients):
    resampling_coefficients = resampling_dispersion_coefficients[:2]
    dispersion_coefficients = resampling_dispersion_coefficients[2:]

    spectra = dc_subtract(spectra)
    spectra = crop_spectra(spectra)
    
    # spectra = k_resample(spectra,resampling_coefficients)
    spectra = dispersion_compensate(spectra,dispersion_coefficients)
    spectra = window_spectra(spectra)
    bscan = np.fft.fft(spectra,axis=0)
    # remove one of the conjugate pairs--the top (inverted) one, by default
    bscan = bscan[bscan.shape[0]//2:,:]
    return bscan


def get_bscan(filename,index):
    frame = get_frame(filename,frame_index=index)
    bscan = spectra_to_bscan(frame)
    return bscan

def sharpness(im):
    """Image sharpness"""
    return np.sum(im**2)/(np.sum(im)**2)

def contrast(im):
    """Image contrast"""
    return (np.max(im)-np.min(im))/(np.max(im)+np.min(im))

def max(im):
    """Image contrast"""
    return np.max(im)


def objective_function(resampling_dispersion_coefficients,spectra,image_quality_function=sharpness):
    bscan = spectra_to_bscan(spectra,resampling_dispersion_coefficients)
    bscan = np.abs(bscan)
    bscan_image_quality_function = image_quality_function(bscan)
    print(1.0/bscan_image_quality_function)
    return 1.0/bscan_image_quality_function # remember this is a minimization algorithm

def optimize_resampling_dispersion_coefficients(spectra,initial_guess = [0.0,0.0,0.0,0.0]):

    initial_guess = [0.0,0.0,-1.2e-08,-7.54e-06]
    # run the optimizer
    result = spo.minimize(objective_function,x0=initial_guess,args=(spectra))

    # get the optimized coefficients
    coefs = result.x
    #coefs= [0.00000000e+00,  0.00000000e+00, -1.40199835e-08, -6.71990805e-12]
    print('Optimized coefs: %s'%coefs)
    # For 2023 data, coefficients are [ 6.61026428e-05 -7.20718161e-02 -1.39001361e-06  5.33739024e-05]
    # For 2021 data, coefficients are [-1.95242853e-08  1.35407215e-11 -6.33964727e-11  8.04882650e-12]
    return coefs

def get_coefficients(filename):
    frame = get_frame(filename,10)
    coefs = optimize_resampling_dispersion_coefficients(frame)
    return coefs


################################# Bulk motion functions ################################################

def make_mask(im,threshold):
    mask = np.zeros(im.shape)
    mask[np.where(im>threshold)] = 1
    return mask

def centers_to_edges(bin_centers):
    # convert an array of bin centers to bin edges, using the mean
    # spacing of the centers to determine bin width

    # check if sorted:
    assert all(bin_centers[1:]>bin_centers[:-1])

    bin_width = np.mean(np.diff(bin_centers))
    half_width = bin_width/2.0
    first_edge = bin_centers[0]-half_width
    last_edge = bin_centers[-1]+half_width
    return np.linspace(first_edge,last_edge,len(bin_centers)+1)

def bin_shift_histogram(vals,bin_centers,resample_factor=1):
    shifts = np.linspace(bin_centers[0]/float(len(bin_centers)),
                          bin_centers[-1]/float(len(bin_centers)),resample_factor)

    n_shifts = len(shifts)
    n_bins = len(bin_centers)

    all_counts = np.zeros((n_shifts,n_bins))
    all_edges = np.zeros((n_shifts,n_bins+1))

    for idx,s in enumerate(shifts):
        edges = centers_to_edges(bin_centers+s)
        all_counts[idx,:],all_edges[idx,:] = np.histogram(vals,edges)

    all_centers = (all_edges[:,:-1]+all_edges[:,1:])/2.0
    all_counts = all_counts/float(resample_factor)
    all_centers = all_centers
    
    return all_counts.T.ravel(),all_centers.T.ravel()

def wrap_into_range(arr,phase_limits=(-np.pi,np.pi)):
    lower,upper = phase_limits
    above_range = np.where(arr>upper)
    below_range = np.where(arr<lower)
    arr[above_range]-=2*np.pi
    arr[below_range]+=2*np.pi
    return arr

def get_phase_jumps(phase_stack,mask,
                    n_bins=16,
                    resample_factor=24,
                    n_smooth=5,polynomial_smoothing=True):

    # Take a stack of B-scan phase arrays, with dimensions
    # (z,x,repeats), and return a bulk-motion corrected
    # version
    #phase_stack = np.transpose(phase_stack,(1,2,0))
    n_depth = phase_stack.shape[0]
    n_fast = phase_stack.shape[1]
    n_reps = phase_stack.shape[2]
    
    d_phase_d_t = np.diff(phase_stack,axis=2)
    d_phase_d_t = wrap_into_range(d_phase_d_t)

    d_phase_d_t = np.transpose(np.transpose(d_phase_d_t,(2,0,1))*mask,(1,2,0))
    bin_edges = np.linspace(-np.pi,np.pi,n_bins)
    
    # The key idea here is from Makita, 2006, where it is well explained. In
    # addition to using the phase mode, we also do bin-shifting, in order to
    # smooth the histogram. 
    b_jumps = np.zeros((d_phase_d_t.shape[1:]))
    bin_counts = np.zeros((d_phase_d_t.shape[1:]))

    for f in range(n_fast):
        valid_idx = np.where(mask[:,f])[0]
        for r in range(n_reps-1):
            vals = d_phase_d_t[valid_idx,f,r]
            
            [counts,bin_centers] = bin_shift_histogram(vals,bin_edges,resample_factor)
            bulk_shift = bin_centers[np.argmax(counts)]
            bin_count = np.max(counts)
            b_jumps[f,r] = bulk_shift
            bin_counts[f,r] = bin_count

    # Now unwrap to prevent discontinuities (although this may not impact complex variance)
    b_jumps = np.unwrap(b_jumps,axis=0)

    return b_jumps

def bulk_motion_correct(phase_stack,mask,
                        n_bins=16,
                        resample_factor=24,
                        n_smooth=5):

    # Take a stack of B-scan phase arrays, with dimensions
    # (z,x,repeats), and return a bulk-motion corrected
    # version

    n_reps = phase_stack.shape[2]

    b_jumps = get_phase_jumps(phase_stack,mask,
                              n_bins=n_bins,
                              resample_factor=resample_factor,
                              n_smooth=n_smooth)

    # Now, subtract b_jumps from phase_stack, not including the first repeat
    # Important: this is happening by broadcasting--it requires that the
    # last two dimensions of phase_stack[:,:,1:] be equal in size to the two
    # dimensions of b_jumps
    out = np.copy(phase_stack)

    errs = []
    for rep in range(1,n_reps):
        # for each rep, the total error is the sum of
        # all previous errors
        err = np.sum(b_jumps[:,:rep],axis=1)
        errs.append(err)
        out[:,:,rep] = out[:,:,rep]-err
    out = wrap_into_range(out)
    return out


def guess_bscan_crop_coords(bscan,padding=20):
    mprof = np.mean(np.abs(bscan),axis=1)
    thresh = 2*np.min(mprof)
    z1 = np.where(mprof>thresh)[0][0]-padding
    z2 = np.where(mprof>thresh)[0][-1]+padding
    return z1,z2

################################## ORG functions #######################################################




################################## File management functions ###########################################
def get_bscan_folder(data_filename,make=True):
    ext = os.path.splitext(data_filename)[1]
    bscan_folder = data_filename.replace(ext,'')+'_bscans'
    
    if make:
        os.makedirs(bscan_folder,exist_ok=True)
    return bscan_folder

def get_png_folder(data_filename,make=True):
    ext = os.path.splitext(data_filename)[1]
    bscan_folder = data_filename.replace(ext,'')+'_bscans'
    png_folder = os.path.join(bscan_folder,'png')
    if make:
        os.makedirs(png_folder,exist_ok=True)
    return png_folder


################################## Diagnostics class ##################################################
class Diagnostics:

    def __init__(self,tag,limit=3):
        if tag.find('_bscans')>-1:
            tag = tag.replace('_bscans/','')
        if tag.find('.unp')>-1:
            tag = tag.replace('.unp','')
            
        self.folder = tag+'_diagnostics'
        os.makedirs(self.folder,exist_ok=True)
        self.limit = limit
        self.dpi = 150
        self.figures = {}
        self.labels = {}
        self.counts = {}
        self.done = []
        self.current_figure = None

    def log(self,title,header,data,fmt,clobber):
        print(title)
        print(header)
        print(fmt%data)
        
    def save(self,figure_handle=None,ignore_limit=False):

        if figure_handle is None:
            figure_handle = self.current_figure
        label = self.labels[figure_handle]
        
        if label in self.done:
            return
        
        subfolder = os.path.join(self.folder,label)
        os.makedirs(subfolder,exist_ok=True)
        index = self.counts[label]

        if index<self.limit or ignore_limit:
            outfn = os.path.join(subfolder,'%s_%05d.png'%(label,index))
            plt.figure(label)
            plt.suptitle(label)
            plt.savefig(outfn,dpi=self.dpi)
            #plt.show()
            self.counts[label]+=1
        else:
            self.done.append(label)
        #plt.close(figure_handle.number)
            

    def figure(self,figsize=(6,6),dpi=100,label=None):
        if label is None:
            label = inspect.currentframe().f_back.f_code.co_name
            
        subfolder = os.path.join(self.folder,label)
        if not label in self.counts.keys():
            self.counts[label] = 0
            os.makedirs(subfolder,exist_ok=True)
        fig = plt.figure(label)
        self.labels[fig] = label
        fig.clear()
        fig.set_size_inches(figsize[0],figsize[1], forward=True)
        #out = plt.figure(figsize=figsize,dpi=dpi)
        self.current_figure = fig
        return fig




if __name__=='__main__':
    # some tests
    coefs = get_coefficients('16_53_25.unp')
