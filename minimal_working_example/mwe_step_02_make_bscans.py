import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os
import functions as blobf

dB_clims = (40,90)

try:
    fn = sys.argv[1]
except:
    print('Please supply the filename at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
    sys.exit()

show_bscans = True

bscan_folder = blobf.get_bscan_folder(fn)
cfg = blobf.get_configuration(fn.replace('.unp','.xml'))
n_vol = cfg['n_vol']
n_slow = cfg['n_slow']

try:
    coefs = np.loadtxt('mapping_dispersion_coefficients.txt')
except FileNotFoundError as fnfe:
    print('Cannot find mapping_dispersion_coefficients.txt')
    print('Please run dispersion compensation step first, and save dispersion coefficients to mapping_dispersion_coefficients.txt')
    sys.exit()
    
for v in range(n_vol):
    for s in range(n_slow):
        spectra = blobf.get_frame(fn,s,v)
        bscan = blobf.spectra_to_bscan(spectra,coefs)
        if n_vol==1:
            outfn = os.path.join(bscan_folder,'complex_%05d.npy'%s)
        elif n_vol>1:
            volume_folder = os.path.join(bscan_folder,'volume_%02d'%v)
            os.makedirs(volume_folder,exist_ok=True)
            outfn = os.path.join(volume_folder,'complex_%05d.npy'%s)

        if show_bscans:
            plt.cla()
            plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray',clim=dB_clims)
            plt.pause(0.00001)

        np.save(outfn,bscan)
        print('Saving volume %d bscan %d to %s'%(v,s,outfn))
