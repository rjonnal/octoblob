from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.optimize as spo

dB_lims = (43,88)

spectra = blobf.load_spectra('spectra_00100.npy')
spectra = blobf.dc_subtract(spectra)

def crop_bscan(bscan,top_crop=300,bottom_crop=50):
    sz,sx = bscan.shape
    bscan = bscan[sz//2:,:]
    # remove dc
    bscan = bscan[top_crop:-bottom_crop,:]
    return bscan

def show_bscan(bscan,iqf,ax=plt.gca()):
    ax.clear()
    ax.imshow(20*np.log10(bscan),cmap='gray',clim=dB_lims)
    ax.set_title(iqf(bscan))
    plt.pause(0.000001)

def bscan_m(mcoefs,spectra):
    spectra = blobf.k_resample(spectra,mcoefs)
    bscan = np.abs(np.fft.fft(spectra,axis=0))
    bscan = crop_bscan(bscan)
    return bscan

def bscan_d(dcoefs,spectra):
    spectra = blobf.dispersion_compensate(spectra,dcoefs)
    bscan = np.abs(np.fft.fft(spectra,axis=0))
    bscan = crop_bscan(bscan)
    return bscan

def bscan_md(mdcoefs,spectra):
    spectra = blobf.k_resample(spectra,mdcoefs[:2])
    spectra = blobf.dispersion_compensate(spectra,mdcoefs[2:])
    bscan = np.abs(np.fft.fft(spectra,axis=0))
    bscan = crop_bscan(bscan)
    return bscan

def obj_m(mcoefs,spectra,show,iqf):
    """Optimize mapping"""
    bscan = bscan_m(mcoefs,spectra)
    if show:
        show_bscan(bscan,iqf)
    return 1.0/iqf(bscan)

def obj_d(dcoefs,spectra,show,iqf):
    """Optimze dispersion"""
    bscan = bscan_d(dcoefs,spectra)
    if show:
        show_bscan(bscan,iqf)
    return 1.0/iqf(bscan)

def obj_md(mdcoefs,spectra,show,iqf):
    """Optimize mapping and dispersion"""
    bscan = bscan_md(mdcoefs,spectra)
    if show:
        show_bscan(bscan,iqf)
    return 1.0/iqf(bscan)


# spo.minimize accepts an additional argument, a dictionary containing further
# options; we want can specify an error tolerance, say about 1% of the bounds.
# we can also specify maximum iterations:
optimization_options = {'xatol':1e-10,'maxiter':1000}

# optimization algorithm:
# See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
method = 'nelder-mead'

mapping_bounds = [(-2e-8,1e-9),(-6e-5,2e-6)]
dispersion_bounds = [(-5e-8,5e-8),(-1e-4,1e-4)]
md_bounds = mapping_bounds+dispersion_bounds


def run(obj_f,bscan_f,spectra,bounds):
    if obj_f==obj_md:
        init = [0.0,0.0,0.0,0.0]
    else:
        init = [0.0,0.0]
    plt.subplot(1,2,1)
    plt.title('%s (pre)'%obj_f.__doc__)
    plt.imshow(bscan_f(spectra,init),cmap='gray',aspect='auto')
    plt.colorbar()
    res = spo.minimize(obj_f,init,args=(spectra,True,blobf.iq_maxes),bounds=bounds,method=method,options=optimization_options)

run(obj_d,bscan_d,spectra,dispersion_bounds)

#res = spo.minimize(obj_d,[0.0,0.0],args=(spectra,True,blobf.iq_max),method=method,options=optimization_options)
#res = spo.minimize(obj_d,[0.0,0.0],args=(spectra,True,blobf.iq_max),method=method,options=optimization_options)

