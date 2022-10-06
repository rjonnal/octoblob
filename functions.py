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
fbg_position = 90
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



# An example of optimizing dispersion:

# First, we need an objective function that takes the two dispersion coefficients and outputs
# a single value to be minimized; for simplicity, we'll use the reciprocal of the brightest
# pixel in the image. An oddity here is that the function can see outside its scope and thus
# has access to the variable 'spectra', defined at the top by loading from the NPY file. We
# then call our process_bscans function, using the coefficients passed into this objective
# function. From the resulting B-scan, we calculate our value to be minimized:
def obj_func(coefs,save=False):
    bscan = process_bscan(spectra,coefs)
    # we don't need the complex conjugate, so let's determine the size of the B-scan and crop
    # the bottom half (sz//2:) for use. (// means integer division--we can't index with floats;
    # also, the sz//2: is implied indexing to the bottom of the B-scan:
    sz,sx = bscan.shape
    bscan = bscan[sz//2:,:]
    # we also want to avoid DC artifacts from dominating the image brightness or gradients,
    # so let's remove the bottom, using negative indexing.
    # See: https://numpy.org/devdocs/user/basics.indexing.html
    bscan = bscan[:-50,:]
    # Finally let's compute the amplitude (modulus) max and return its reciprocal:
    bscan = np.abs(bscan)
    bscan = bscan[-300:] # IMPORTANT--THIS WON'T WORK IN GENERAL, ONLY ON THIS DATA SET 16_53_25
    out = 1.0/np.max(bscan)
    
    # Maybe we want to visualize it; change to False to speed things up
    if True:
        # clear the current axis
        plt.cla()
        # show the image:
        plt.imshow(20*np.log10(bscan),cmap='gray',clim=dB_lims)
        # pause:
        plt.pause(0.001)

    if save:
        order = len(coefs)+1
        os.makedirs('dispersion_compensation_results',exist_ok=True)
        plt.cla()
        plt.imshow(20*np.log10(bscan),cmap='gray',clim=dB_lims)
        plt.title('order %d\n %s'%(order,list(coefs)+[0.0,0.0]),fontsize=10)
        plt.colorbar()
        plt.savefig('dispersion_compensation_results/order_%d.png'%order,dpi=150)
    return out


# Now we can define some settings for the optimization:

def optimize_dispersion(spectra,obj_func,initial_guess):

    # spo.minimize accepts an additional argument, a dictionary containing further
    # options; we want can specify an error tolerance, say about 1% of the bounds.
    # we can also specify maximum iterations:
    optimization_options = {'xatol':1e-10,'maxiter':10000}

    # optimization algorithm:
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    method = 'nelder-mead'


    # Now we run it; Nelder-Mead cannot use bounds, so we pass None
    res = spo.minimize(obj_func,initial_guess,method='nelder-mead',bounds=None,options=optimization_options)

    print('Optimization result (order: %d):'%order)
    print(res.x)
    print(obj_func(res.x,save=True))
