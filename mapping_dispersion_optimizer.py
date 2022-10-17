from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.optimize as spo
import sys
import numba

dB_lims = (45,80)
fbg_search_distance = 11

def show_bscan(ax,bscan,iqf=blobf.sharpness):
    print(ax)
    sys.exit()
    #ax.clear()
    ax.imshow(20*np.log10(bscan),cmap='gray',clim=dB_lims)
    ax.set_title(iqf(bscan))
    plt.pause(0.000001)

def bscan_m(mcoefs,spectra):
    spectra = blobf.k_resample(spectra,mcoefs)
    bscan = np.abs(np.fft.fft(spectra,axis=0))
    bscan = blobf.crop_bscan(bscan)
    return bscan

def bscan_d(dcoefs,spectra):
    spectra = blobf.dispersion_compensate(spectra,dcoefs)
    bscan = np.abs(np.fft.fft(spectra,axis=0))
    bscan = blobf.crop_bscan(bscan)
    return bscan

def bscan_md(mdcoefs,spectra):
    spectra = blobf.k_resample(spectra,mdcoefs[:2])
    spectra = blobf.dispersion_compensate(spectra,mdcoefs[2:])
    bscan = np.abs(np.fft.fft(spectra,axis=0))
    bscan = blobf.crop_bscan(bscan)
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
    print(np.random.rand())
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



def run(spectra,show=False):
    #mapping_bounds = [(-2e-8,1e-9),(-6e-5,2e-6)]
    mapping_bounds = [(None,None),(None,None)]
    #dispersion_bounds = [(-5e-8,5e-8),(-1e-4,1e-4)]
    dispersion_bounds = [(None,None),(None,None)]

    bounds = mapping_bounds+dispersion_bounds
    
    obj_f=obj_md
    bscan_f=bscan_md
    init = [0.0,0.0,0.0,0.0]

    if show:
        fig = plt.figure(figsize=(8,8))
        ax1,ax2 = fig.subplots(1,2)
        print(ax1,ax2)
        sys.exit()
        show_bscan(ax1,bscan_f(init,spectra))
        plt.pause(0.0001)
        
    res = spo.minimize(obj_f,init,args=(spectra,False,blobf.sharpness),bounds=bounds,method=method,options=optimization_options)

    if show:
        plt.subplot(1,2,2)
        plt.title('%s (post)'%obj_f.__doc__)
        show_bscan(ax2,bscan_f(res.x,spectra))
        plt.show()

    return res.x
