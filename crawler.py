from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy.optimize as spo
import sys,os
import pathlib

dB_lims = (45,80)
fbg_search_distance = 11
noise_samples = 80
output_folder = 'crawler_output'

unp_files = pathlib.Path('.').rglob('*.unp')

os.makedirs(output_folder,exist_ok=True)
for unp_file in unp_files:
    print(unp_file)
    path,fn = os.path.split(unp_file)
    path = os.path.join(output_folder,path)
    fn_root = fn.replace('.unp','')+'_diagnostics'
    path = os.path.join(path,fn_root)
    
    os.makedirs(path,exist_ok=True)
    print(unp_file)
    spectra = blobf.load_spectra(unp_file,0)
    spectra = blobf.fbg_align(spectra,fbg_search_distance,noise_samples=noise_samples,diagnostics_path=path)
    
    
sys.exit()




try:
    fn = sys.argv[1]
    index = int(sys.argv[2])
    spectra = blobf.load_spectra(fn,index)
except IndexError:
    #spectra = blobf.load_spectra('spectra_00100.npy')
    #spectra = blobf.load_spectra('../examples/oct_processing/test_1.unp',20)
    #spectra = blobf.load_spectra('/media/nas_data/conventional_org/flash/2022.08.24_jonnal_0013/2deg/11_33_26.unp',20)
    spectra = blobf.load_spectra('/media/nas_data/conventional_org/flash/2022.09.23_jonnal_0014/2deg/15_19_21.unp',20)


spectra = blobf.fbg_align(spectra,fbg_search_distance)
spectra = blobf.dc_subtract(spectra)

def crop_bscan(bscan,top_crop=300,bottom_crop=50):
    sz,sx = bscan.shape
    bscan = bscan[sz//2:,:]
    # remove dc
    bscan = bscan[top_crop:-bottom_crop,:]
    return bscan

def show_bscan(ax,bscan,iqf=blobf.sharpness):
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
mapping_bounds = [(None,None),(None,None)]
dispersion_bounds = [(-5e-8,5e-8),(-1e-4,1e-4)]
dispersion_bounds = [(None,None),(None,None)]

md_bounds = mapping_bounds+dispersion_bounds


def run(obj_f,bscan_f,spectra,bounds):
    if obj_f==obj_md:
        init = [0.0,0.0,0.0,0.0]
    else:
        init = [0.0,0.0]
    fig = plt.figure(figsize=(8,8))
    ax1,ax2 = fig.subplots(1,2)
    #ax1.set_title('%s (pre)'%obj_f.__doc__)
    show_bscan(ax1,bscan_f(init,spectra))
    plt.pause(0.0001)
    res = spo.minimize(obj_f,init,args=(spectra,False,blobf.sharpness),bounds=bounds,method=method,options=optimization_options)
    plt.subplot(1,2,2)
    plt.title('%s (post)'%obj_f.__doc__)
    show_bscan(ax2,bscan_f(res.x,spectra))
    plt.show()

    
run(obj_md,bscan_md,spectra,dispersion_bounds)

#res = spo.minimize(obj_d,[0.0,0.0],args=(spectra,True,blobf.iq_max),method=method,options=optimization_options)
#res = spo.minimize(obj_d,[0.0,0.0],args=(spectra,True,blobf.iq_max),method=method,options=optimization_options)

