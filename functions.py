import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as spo
import sys,os,glob,logging
import scipy.interpolate as spi
import scipy.signal as sps

###################################################
# Simplified OCT functions for exporatory analysis,
# REPL applications, and illustration
###################################################

#######################################
## Constants here--adjust as necessary:

dB_lims = [45,85]
crop_height = 300 # height of viewable B-scan, centered at image z centroid (center of mass)

# step sizes for incrementing/decrementing coefficients:
mapping_steps = [1e-4,1e-2]
dispersion_steps = [1e-10,1e-8]
# let's start doing this explicitly with a function in this module, instead of buried inside
# the OCTRawData class; fbg_position used to be set to 90; now set it to None and handle it separately
fbg_position = None
bit_shift_right = 4
window_sigma = 0.9

k_crop_1 = 100
k_crop_2 = 1490


#######################################

# Now we'll define some functions for the half-dozen or so processing
# steps:

def get_source(fn,diagnostics=None):
    from octoblob.data_source import DataSource
    #import octoblob as blob
    print(fn)
    src = DataSource(fn)
    return src

def crop_spectra(spectra,diagnostics=None):
    if not diagnostics is None:
        fig = diagnostics.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(np.abs(spectra),aspect='auto')
        plt.title('pre cropping')
    spectra = spectra[k_crop_1:k_crop_2,:]
    
    if not diagnostics is None:
        plt.subplot(1,2,2)
        plt.imshow(np.abs(spectra),aspect='auto')
        plt.title('post cropping')
        diagnostics.save(fig)
        
    return spectra

def load_spectra(fn,index=0):
    ext = os.path.splitext(fn)[1]
    if ext.lower()=='.unp':
        src = get_source(fn)
        
        index = index%(n_slow*n_vol)
        spectra = src.get_frame(index)
    elif ext.lower()=='.npy':
        spectra = np.load(fn)
    else:
        sys.exit('File %s is of unknown type.'%fn)
    return spectra.astype(np.float)


def fbg_align(spectra,fbg_search_distance=15,noise_samples=80,diagnostics=None):
    if not diagnostics is None:
        fig = diagnostics.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(spectra),aspect='auto')
        
    spectra[:noise_samples,:] = spectra[noise_samples,:]
    prof = np.nanmean(spectra,axis=1)
    idx = np.argmax(np.diff(prof))

    fbg_locations = np.zeros(spectra.shape[1],dtype=np.int)
    temp = np.zeros(len(prof)-1)

    for k in range(spectra.shape[1]):
        temp[:] = np.diff(spectra[:,k],axis=0)
        temp[:idx-fbg_search_distance] = 0
        temp[idx+fbg_search_distance:] = 0
        fbg_locations[k] = np.argmax(temp[:])

    fbg_locations = fbg_locations - int(np.median(fbg_locations))
    for k in range(spectra.shape[1]):
        spectra[:,k] = np.roll(spectra[:,k],-fbg_locations[k])

    if not diagnostics is None:
        plt.subplot(1,2,2)
        plt.imshow(np.abs(spectra),aspect='auto')
        diagnostics.save(fig)
        
    return spectra

# We need a way to estimate and remove DC:
def dc_subtract(spectra,diagnostics=None):
    """Estimate DC by averaging spectra spatially (dimension 1),
    then subtract by broadcasting."""
    if not diagnostics is None:
        fig = diagnostics.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(spectra),aspect='auto')
        
    dc = spectra.mean(1)
    # Do the subtraction by array broadcasting, for efficiency.
    # See: https://numpy.org/doc/stable/user/basics.broadcasting.html
    out = (spectra.T-dc).T
    if not diagnostics is None:
        plt.subplot(1,2,2)
        plt.imshow(out,aspect='auto')
        diagnostics.save(fig)
    return out


# Next we need a way to adjust the values of k at each sample, and then
# interpolate into uniformly sampled k:
def k_resample(spectra,coefficients,diagnostics=None):
    """Resample the spectrum such that it is uniform w/r/t k.
    Notes:
      1. The coefficients here are for a polynomial defined on
         pixels, so they're physically meaningless. It would be
         better to define our polynomials on k, because then
         we could more easily quantify and compare the chirps
         of multiple light sources, for instance. Ditto for the
         dispersion compensation code.
    """
    if not any(coefficients):
        return spectra

    if not diagnostics is None:
        fig = diagnostics.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(spectra),aspect='auto')
        
    coefficients = coefficients + [0.0,0.0]
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
    if not diagnostics is None:
        plt.subplot(1,2,2)
        plt.imshow(interpolated,aspect='auto')
        diagnostics.save(fig)
        
    return interpolated

# Next we need to dispersion compensate; for historical reasons the correction polynomial
# is defined on index x rather than k, but for physically meaningful numbers we should
# use k instead
def dispersion_compensate(spectra,coefficients,diagnostics=None):
    if not any(coefficients):
        return spectra

    if not diagnostics is None:
        fig = diagnostics.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(spectra),aspect='auto')

    coefs = list(coefficients) + [0.0,0.0]
    # define index x:
    x = np.arange(1,spectra.shape[0]+1)
    # define the phasor and multiply by spectra using broadcasting:
    dechirping_phasor = np.exp(-1j*np.polyval(coefs,x))
    dechirped = (spectra.T*dechirping_phasor).T
    if not diagnostics is None:
        plt.subplot(1,2,2)
        plt.imshow(np.abs(spectra),aspect='auto')
        diagnostics.save(fig)
        
    return dechirped


# Next we multiply the spectra by a Gaussian window, in order to reduce ringing
# in the B-scan due to edges in the spectra:
def gaussian_window(spectra,sigma,diagnostics=None):
    if sigma>1e5:
        return spectra
    
    if not diagnostics is None:
        fig = diagnostics.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(spectra),aspect='auto')
        
    # Define a Gaussian window with passed sigma
    x = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma**2))
    # Multiply spectra by window using broadcasting:
    out = (spectra.T*x).T

    if not diagnostics is None:
        plt.subplot(1,2,2)
        plt.imshow(np.abs(spectra),aspect='auto')
        diagnostics.save(fig)
        
    return out


# # Now let's define a processing function that takes the spectra and two dispersion coefficients
# # and produces a B-scan:
# def process_bscan(spectra,mapping_coefficients=[0.0],dispersion_coefficients=[0.0],window_sigma=0.9):
#     spectra = dc_subtract(spectra)
#     # When we call dispersion_compensate, we have to pass the c3 and c2 coefficients as well as
#     # two 0.0 values, to make clear that we want orders 3, 2, 1, 0. This enables us to use the
#     # polyval function of numpy instead of writing the polynomial ourselves, e.g. c3*x**3+c2*x**x**2,
#     # since the latter is more likely to cause bugs.
#     spectra = k_resample(spectra,mapping_coefficients)
#     spectra = dispersion_compensate(spectra,dispersion_coefficients)
#     spectra = gaussian_window(spectra,sigma=window_sigma)
#     bscan = np.fft.fft(spectra,axis=0)
#     return bscan


# Image quality metrics
def iq_max(im):
    """Image max"""
    return np.max(im)

def iq_maxes(im):
    """Mean of brightest\n1 pct of pixels"""
    temp = im.ravel()
    N = round(len(temp)*0.01)
    temp = np.partition(-temp, N)
    result = -temp[:N]
    return np.mean(result)

def gradient_mean(im):
    """Mean of absolute\nz-derivative"""
    return np.mean(np.abs(np.diff(im,axis=0)))

def gradient_median(im):
    """Median of absolute\nz-derivative"""
    return np.mean(np.abs(np.diff(im,axis=0)))

def average_aline_contrast(im):
    """Mean of A-scan\nMichelson contrast""" 
    x = np.max(im,axis=0)
    n = np.min(im,axis=0)
    return np.mean((x-n)/(x+n))

def sharpness(im):
    """Image sharpness"""
    return np.sum(im**2)/(np.sum(im)**2)

def center_sharpness(im,fraction=0.5):
    """Image sharpness"""
    sy,sx = im.shape
    mid = sy//2
    x1 = mid-round(sx*0.5*fraction)
    x2 = mid+round(sx*0.5*fraction)
    return np.sum(im[:,x1:x2]**2)/(np.sum(im[:,x1:x2])**2)

def crop_bscan(bscan,top_crop=350,bottom_crop=30,diagnostics=None):
    sz,sx = bscan.shape
    bscan = bscan[sz//2:,:]
    bscan = bscan[top_crop:-bottom_crop,:]
    return bscan

def dB(arr):
    return 20*np.log10(np.abs(arr))


def threshold_mask(arr,threshold):
    out = np.zeros(arr.shape)
    out[np.where(arr>threshold)] = 1.0
    return out

def percentile_mask(arr,percentile_threshold):
    threshold = np.percentile(arr,percentile_threshold)
    return threshold_mask(arr,threshold)

def spectra_to_bscan(mdcoefs,spectra,diagnostics=None):
    spectra = fbg_align(spectra,diagnostics=diagnostics)
    spectra = dc_subtract(spectra,diagnostics=diagnostics)
    spectra = crop_spectra(spectra,diagnostics=diagnostics)
    
    if diagnostics is not None:
        fig = diagnostics.figure(figsize=(6,4))
        plt.subplot(2,2,1)
        plt.imshow(dB(crop_bscan(np.fft.fft(spectra,axis=0))),aspect='auto',clim=(45,85),cmap='gray')
        plt.title('raw B-scan')
            
    spectra = k_resample(spectra,mdcoefs[:2],diagnostics=None)

    if diagnostics is not None:
        plt.subplot(2,2,2)
        plt.imshow(dB(crop_bscan(np.fft.fft(spectra,axis=0))),aspect='auto',clim=(45,85),cmap='gray')
        plt.title('after k-resampling')

    spectra = dispersion_compensate(spectra,mdcoefs[2:],diagnostics=None)

    if diagnostics is not None:
        plt.subplot(2,2,3)
        plt.imshow(dB(crop_bscan(np.fft.fft(spectra,axis=0))),aspect='auto',clim=(45,85),cmap='gray')
        plt.title('after dispersion_compensation')

    spectra = gaussian_window(spectra,sigma=0.9,diagnostics=None)
    
    if diagnostics is not None:
        plt.subplot(2,2,4)
        plt.imshow(dB(crop_bscan(np.fft.fft(spectra,axis=0))),aspect='auto',clim=(45,85),cmap='gray')
        plt.title('after windowing')
        diagnostics.save(fig)
        
    bscan = np.fft.fft(spectra,axis=0)
    bscan = crop_bscan(bscan)
    return bscan

def flatten_volume(folder,nref=3,diagnostics=None):
    flist = glob.glob(os.path.join(folder,'*.npy'))
    flist.sort()
    N = len(flist)
    
    # grab a section from the middle of the volume to use as a reference
    ref_size = nref
    ref_flist = flist[N//2-ref_size//2:N//2+ref_size//2+1]
    ref = np.abs(np.load(ref_flist[0])).astype(np.float)
    for f in ref_flist[1:]:
        ref = ref + np.abs(np.load(f)).astype(np.float)
    ref = ref/float(ref_size)
    ref = np.mean(ref,axis=1)

    coefs = []
    shifts = []

    out_folder = os.path.join(folder,'flattened')
    os.makedirs(out_folder,exist_ok=True)

    pre_corrected_fast_projection = []
    post_corrected_fast_projection = []

    for f in flist:
        tar_bscan = np.load(f)
        tar = np.mean(np.abs(tar_bscan).astype(np.float),axis=1)

        pre_corrected_fast_projection.append(tar)
        
        num = np.fft.fft(tar)*np.conj(np.fft.fft(ref))
        denom = np.abs(num)
        nxc = np.real(np.fft.ifft(num/denom))
        shift = np.argmax(nxc)
        if shift>len(nxc)//2:
            shift = shift-len(nxc)
        shifts.append(shift)
        coefs.append(np.max(nxc))
        logging.info('flatten_volume cross-correlating file %s'%f)
        

    shifts = np.array(shifts)
    shifts = sps.medfilt(shifts,3)
    shifts = np.round(-shifts).astype(np.int)
    
    for f,shift in zip(flist,shifts):
        tar_bscan = np.load(f)
        tar_bscan = np.roll(tar_bscan,shift,axis=0)

        proj = np.mean(np.abs(tar_bscan).astype(np.float),axis=1)
        post_corrected_fast_projection.append(proj)
        logging.info('flatten_volume rolling file %s by %d'%(f,shift))
        out_fn = os.path.join(out_folder,os.path.split(f)[1])
        np.save(out_fn,tar_bscan)


    if diagnostics is not None:
        pre_corrected_fast_projection = np.array(pre_corrected_fast_projection).T
        post_corrected_fast_projection = np.array(post_corrected_fast_projection).T
        fig = diagnostics.figure(figsize=(9,3))
        ax1,ax2,ax3 = fig.subplots(1,3)
        ax1.imshow(pre_corrected_fast_projection,aspect='auto',cmap='gray')
        ax2.imshow(post_corrected_fast_projection,aspect='auto',cmap='gray')
        ax3.plot(np.mean(pre_corrected_fast_projection,axis=1),label='pre')
        ax3.plot(np.mean(post_corrected_fast_projection,axis=1),label='post')
        ax3.legend()
        diagnostics.save(fig)

