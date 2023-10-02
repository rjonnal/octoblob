import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os,glob
import functions as blobf

do_post_processing_crop = True

dB_clims = (40,90)

try:
    fn = sys.argv[1]
except:
    print('Please supply the filename at the command line and, optionally, the indices of the first and last B-scans to process, i.e., python mweXXX.py XX_YY_ZZ.unp 80 120')
    sys.exit()

try:
    start_bscan = int(sys.argv[2])
    end_bscan = int(sys.argv[3])
except Exception as e:
    start_bscan = 80
    end_bscan = 130
    
show_bscans = True

bscan_folder = blobf.get_bscan_folder(fn)
png_folder = blobf.get_png_folder(fn)

cfg = blobf.get_configuration(fn.replace('.unp','.xml'))
n_vol = cfg['n_vol']
n_slow = cfg['n_slow']

try:
    coefs = np.loadtxt('mapping_dispersion_coefficients.txt')
except FileNotFoundError as fnfe:
    print('Cannot find mapping_dispersion_coefficients.txt')
    print('Please run dispersion compensation step first, and save dispersion coefficients to mapping_dispersion_coefficients.txt')
    sys.exit()

magnitudes = []


for v in range(n_vol):
    for s in range(n_slow):
        if s<start_bscan or s>=end_bscan:
            continue
        spectra = blobf.get_frame(fn,s,v)
        bscan = blobf.spectra_to_bscan(spectra,coefs)
        if n_vol==1:
            outfn = os.path.join(bscan_folder,'complex_%05d.npy'%s)
        elif n_vol>1:
            volume_folder = os.path.join(bscan_folder,'volume_%02d'%v)
            os.makedirs(volume_folder,exist_ok=True)
            outfn = os.path.join(volume_folder,'complex_%05d.npy'%s)

        if show_bscans:
            plt.subplot(1,2,1)
            plt.cla()
            plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray',clim=dB_clims)
            plt.subplot(1,2,2)
            plt.cla()
            plt.plot(np.mean(np.abs(bscan),axis=1))
            
            png_fn = os.path.join(png_folder,'amplitude_%05d.png'%s)
            plt.savefig(png_fn)
            plt.pause(0.00001)

        if do_post_processing_crop:
            magnitudes.append(np.abs(bscan))

        np.save(outfn,bscan)
        print('Saving volume %d bscan %d to %s'%(v,s,outfn))

    if show_bscans:
        plt.close()

        
if do_post_processing_crop:
    bscan_mean = np.mean(magnitudes,axis=0)
    plt.imshow(20*np.log10(bscan_mean))
    plt.title('Note z1 and z2 cropping coordinates.')
    plt.show()
    z1 = int(input('Please enter the inner cropping coordinate (z1): '))
    z2 = int(input('Please enter the outer cropping coordinate (z2): '))
    bscan_flist = glob.glob(os.path.join(bscan_folder,'complex*.npy'))
    for bf in bscan_flist:
        print('Cropping %s'%bf)
        bscan = np.load(bf)
        bscan = bscan[z1:z2,:]
        np.save(bf,bscan)
