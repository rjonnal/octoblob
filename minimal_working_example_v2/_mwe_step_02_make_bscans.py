import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os,glob
import functions as blobf
from config import autocrop,use_generic_mapping_dispersion_file,start_bscan,end_bscan,dB_clims,show_bscans,left_crop,right_crop,oversample

try:
    fn = sys.argv[1]
except:
    print('Please supply the filename at the command line and, optionally, the indices of the first and last B-scans to process, i.e., python mweXXX.py XX_YY_ZZ.unp')
    sys.exit()

    
bscan_folder = blobf.get_bscan_folder(fn)
png_folder = blobf.get_png_folder(fn)

cfg = blobf.get_configuration(fn.replace('.unp','.xml'))
n_vol = cfg['n_vol']
n_slow = cfg['n_slow']

tag = fn.replace('.unp','')

try:
    coefs = np.loadtxt('%s_mapping_dispersion_coefficients.txt'%tag)
    print('Using %s_mapping_dispersion_coefficients.txt for dispersion compensation.'%tag)
except FileNotFoundError as fnfe:
    print('Cannot find %s_mapping_dispersion_coefficients.txt'%tag)
    if use_generic_mapping_dispersion_file:
        try:
            coefs = np.loadtxt('mapping_dispersion_coefficients.txt')
            print('Using generic mapping_dispersion_coefficients.txt for dispersion compensation.')
        except FileNotFoundError:
            print('Please run dispersion compensation step first, and save dispersion coefficients to %s_mapping_dispersion_coefficients.txt'%tag)
            sys.exit()

bscans = []

for v in range(n_vol):
    for s in range(n_slow):
        if s<start_bscan or s>=end_bscan:
            continue
        spectra = blobf.get_frame(fn,s,v)
        if oversample>1:
            sy,sx = spectra.shape
            new_spectra = np.zeros((sy*oversample,sx))
            new_spectra[:sy,:] = spectra
            spectra = new_spectra
        bscan = blobf.spectra_to_bscan(spectra,coefs)
        bscan = np.abs(bscan)
        bscans.append(bscan)
        
bscans = np.array(bscans)
mbscan = np.mean(np.abs(bscans),axis=0)
z1,z2 = blobf.guess_bscan_crop_coords(mbscan)

for v in range(n_vol):
    for s in range(n_slow):
        if start_bscan is not None:
            if s<start_bscan:
                continue
        if end_bscan is not None:
            if s>=end_bscan:
                continue
        if s<start_bscan or s>=end_bscan:
            continue
        spectra = blobf.get_frame(fn,s,v)
        bscan = blobf.spectra_to_bscan(spectra,coefs)
        if autocrop:
            bscan = bscan[z1:z2,left_crop:-right_crop]
            
        if n_vol==1:
            outfn = os.path.join(bscan_folder,'complex_%05d.npy'%s)
        elif n_vol>1:
            volume_folder = os.path.join(bscan_folder,'volume_%02d'%v)
            os.makedirs(volume_folder,exist_ok=True)
            outfn = os.path.join(volume_folder,'complex_%05d.npy'%s)

        if show_bscans:
            plt.cla()
            plt.imshow(blobf.dB(bscan),cmap='gray',clim=dB_clims)
            
            png_fn = os.path.join(png_folder,'amplitude_%05d.png'%s)
            plt.savefig(png_fn)
            plt.pause(0.00001)

        np.save(outfn,bscan)
        print('Saving volume %d bscan %d to %s'%(v,s,outfn))

    if show_bscans:
        plt.close()

        
