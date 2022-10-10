import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as spo
import sys,os,glob
import scipy.interpolate as spi

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
#######################################

# Now we'll define some functions for the half-dozen or so processing
# steps:
def load_spectra(fn,index=0):
    ext = os.path.splitext(fn)[1]
    if ext.lower()=='.unp':
        from octoblob import config_reader
        import octoblob as blob
        
        cfg = config_reader.get_configuration(fn.replace('.unp','.xml'))
        n_vol = cfg['n_vol']
        n_slow = cfg['n_slow']
        n_repeats = cfg['n_bm_scans']
        n_fast = cfg['n_fast']
        n_depth = cfg['n_depth']

        # some conversions to comply with old conventions:
        n_slow = n_slow//n_repeats
        n_fast = n_fast*n_repeats

        
        src = blob.OCTRawData(fn,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,bit_shift_right=bit_shift_right)

        index = index%(n_slow*n_vol)
        spectra = src.get_frame(index)
    elif ext.lower()=='.npy':
        spectra = np.load(fn)
    else:
        sys.exit('File %s is of unknown type.'%fn)
    return spectra.astype(np.float)


def fbg_align(spectra,fbg_search_distance,noise_samples=80,diagnostics_path=None):
    spectra[:noise_samples,:] = spectra[noise_samples,:]
    prof = np.nanmean(spectra,axis=1)
    idx = np.argmax(np.diff(prof))

    fbg_locations = np.zeros(spectra.shape[1],dtype=np.int)
    temp = np.zeros(len(prof)-1)

    if diagnostics_path is not None:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(spectra,aspect='auto',cmap='gray')
        plt.ylim((150,0))
        
    for k in range(spectra.shape[1]):
        temp[:] = np.diff(spectra[:,k],axis=0)
        temp[:idx-fbg_search_distance] = 0
        temp[idx+fbg_search_distance:] = 0
        fbg_locations[k] = np.argmax(temp)

    fbg_locations = fbg_locations - int(np.median(fbg_locations))
    for k in range(spectra.shape[1]):
        spectra[:,k] = np.roll(spectra[:,k],-fbg_locations[k])

    if diagnostics_path is not None:
        plt.subplot(1,2,2)
        plt.imshow(spectra,aspect='auto',cmap='gray')
        plt.ylim((150,0))
        plt.savefig(os.path.join(diagnostics_path,'fbg_align.png'))
        plt.close()

    return spectra

# We need a way to estimate and remove DC:
def dc_subtract(spectra):
    """Estimate DC by averaging spectra spatially (dimension 1),
    then subtract by broadcasting."""
    dc = spectra.mean(1)
    # Do the subtraction by array broadcasting, for efficiency.
    # See: https://numpy.org/doc/stable/user/basics.broadcasting.html
    out = (spectra.T-dc).T
    return out


# Next we need a way to adjust the values of k at each sample, and then
# interpolate into uniformly sampled k:
def k_resample(spectra,coefficients):
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
    return interpolated

# Next we need to dispersion compensate; for historical reasons the correction polynomial
# is defined on index x rather than k, but for physically meaningful numbers we should
# use k instead
def dispersion_compensate(spectra,coefficients):
    if not any(coefficients):
        return spectra
    coefs = list(coefficients) + [0.0,0.0]
    # define index x:
    x = np.arange(1,spectra.shape[0]+1)
    # define the phasor and multiply by spectra using broadcasting:
    dechirping_phasor = np.exp(-1j*np.polyval(coefs,x))
    dechirped = (spectra.T*dechirping_phasor).T
    return dechirped


# Next we multiply the spectra by a Gaussian window, in order to reduce ringing
# in the B-scan due to edges in the spectra:
def gaussian_window(spectra,sigma):
    if sigma>1e5:
        return spectra
    # Define a Gaussian window with passed sigma
    x = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma**2))
    # Multiply spectra by window using broadcasting:
    out = (spectra.T*x).T
    return out


# Now let's define a processing function that takes the spectra and two dispersion coefficients
# and produces a B-scan:
def process_bscan(spectra,mapping_coefficients=[0.0],dispersion_coefficients=[0.0],window_sigma=0.9):
    spectra = dc_subtract(spectra)
    # When we call dispersion_compensate, we have to pass the c3 and c2 coefficients as well as
    # two 0.0 values, to make clear that we want orders 3, 2, 1, 0. This enables us to use the
    # polyval function of numpy instead of writing the polynomial ourselves, e.g. c3*x**3+c2*x**x**2,
    # since the latter is more likely to cause bugs.
    spectra = k_resample(spectra,mapping_coefficients)
    spectra = dispersion_compensate(spectra,dispersion_coefficients)
    spectra = gaussian_window(spectra,sigma=window_sigma)
    bscan = np.fft.fft(spectra,axis=0)
    return bscan


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

