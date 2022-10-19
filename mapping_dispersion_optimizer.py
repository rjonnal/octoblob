from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.optimize as spo
import sys
import numba
import logging
from time import sleep

dB_clim = (45,85)
fbg_search_distance = 11

def dB(arr):
    return 20*np.log10(np.abs(arr))

def obj_md(mdcoefs,spectra,bscan_function,iqf,ax=None,verbose=False):
    """Optimize mapping and dispersion"""
    bscan = bscan_function(mdcoefs,spectra)
    bscan = np.abs(bscan)
    iq = iqf(bscan)
    if ax is not None:
        ax.clear()
        show_bscan(ax,bscan)
        plt.pause(0.0000001)
    if verbose:
        logging.info('Coefs %s -> %s value of %0.1e.'%(mdcoefs,iqf.__doc__,iq))
    else:
        sys.stdout.write('.')
        sys.stdout.flush()
        sleep(0.0001)
    return 1.0/iq

def show_bscan(ax,bscan):
    ax.imshow(dB(bscan),cmap='gray',clim=dB_clim)

def optimize(spectra,bscan_function,show=False,verbose=False,maxiters=200,diagnostics=None):

    # confused about bounds--documentation says they can be used with Nelder-Mead, but warnings
    # say that they can't
    mapping_bounds = [(-2e-8,1e-9),(-6e-5,2e-6)]
    dispersion_bounds = [(-5e-8,5e-8),(-1e-4,1e-4)]
    bounds = mapping_bounds+dispersion_bounds
    #bounds = None
    
    # spo.minimize accepts an additional argument, a dictionary containing further
    # options; we want can specify an error tolerance, say about 1% of the bounds.
    # we can also specify maximum iterations:
    optimization_options = {'xatol':1e-6,'maxiter':200,'disp':False}

    # optimization algorithm:
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    method = 'nelder-mead'

    init = [0.0,0.0,0.0,0.0]

    if diagnostics is not None:
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(blobf.dB(bscan_function(init,spectra)),aspect='auto',clim=(45,85),cmap='gray')
        
    realtime_axis = None
    sys.stdout.write('Optimizing ')
        
    res = spo.minimize(obj_md,init,args=(spectra,bscan_function,blobf.sharpness,realtime_axis,verbose),bounds=bounds,method=method,options=optimization_options)
    
    if diagnostics is not None:
        plt.subplot(1,2,2)
        plt.imshow(blobf.dB(bscan_function(res.x,spectra)),aspect='auto',clim=(45,85),cmap='gray')
        diagnostics.save(fig)
        
    return res.x


def multi_optimize(spectra_list,bscan_function,show_all=False,show_final=False,verbose=False,maxiters=200,diagnostics=None):
    results_coefficients = []
    results_iq = []
    
    for spectra in spectra_list:
        coefs = optimize(spectra,bscan_function,show=show_all,verbose=verbose,maxiters=maxiters,diagnostics=diagnostics)
        results_coefficients.append(coefs)
        iq = obj_md(coefs,spectra,bscan_function,blobf.sharpness)
        results_iq.append(iq)

    winner = np.argmin(results_iq)
    logging.info(results_iq)
    logging.info('winner is index %d'%winner)

    
    for rc,riq in zip(results_coefficients,results_iq):
        logging.info(rc,riq)

    if diagnostics is not None:
        for idx,(spectra,coefs,iq) in enumerate(zip(spectra_list,results_coefficients,results_iq)):
            logging.info('iq from optimization: %0.3f'%iq)
            logging.info('iq from obj_md: %0.3f'%obj_md(coefs,spectra,bscan_function,blobf.sharpness))
            sfig = plt.figure()
            sax = sfig.add_subplot(1,1,1)
            show_bscan(sax,bscan_function(coefs,spectra))
            if idx==winner:
                plt.title('winner %0.3f'%obj_md(coefs,spectra,bscan_function,blobf.sharpness))
            else:
                plt.title('loser %0.3f'%obj_md(coefs,spectra,bscan_function,blobf.sharpness))
            diagnostics.save(sfig,ignore_limit=True)

    return results_coefficients[winner]


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
