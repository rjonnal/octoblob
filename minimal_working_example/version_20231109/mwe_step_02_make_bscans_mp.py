import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os,glob
import functions as blobf
from config import autocrop,use_generic_mapping_dispersion_file,start_bscan,end_bscan
from config import dB_clims,show_bscans,left_crop,right_crop,save_bscan_pngs
from config import require_multiprocessing
import multiprocessing as mp

if __name__=='__main__':

    n_cpus = os.cpu_count()
    
    try:
        fn = sys.argv[1]
    except:
        print('Please supply the filename at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
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

    def get_bscan_and_indices(tup):
        fn = tup[0]
        s = tup[1]
        v = tup[2]
        print('Computing bscan %d volume %d.'%(s,v))
        spectra = blobf.get_frame(fn,s,v)
        bscan = blobf.spectra_to_bscan(spectra,coefs)
        return (bscan,(fn,s,v))
    
    sv_pairs = []
    
    for v in range(n_vol):
        for s in range(n_slow):
            if s<start_bscan or s>end_bscan:
                continue
            sv_pairs.append((fn,s,v))


    try:
        p = mp.Pool(n_cpus)
        bscans_and_indices = p.map(get_bscan_and_indices,sv_pairs)
    except Exception as e:
        print(e)
        if require_multiprocessing:
            sys.exit('Multiprocessing failed. Serial processing aborted because require_multiprocessing=True.')
        bscans_and_indices = []
        for fn,s,v in sv_pairs:
            bscans_and_indices.append(get_bscan_and_indices((fn,s,v)))

    bscans = [tup[0] for tup in bscans_and_indices]
    indices = [tup[1] for tup in bscans_and_indices]
    
    bscans = np.array(bscans)
    mbscan = np.mean(np.abs(bscans),axis=0)
    z1,z2 = blobf.guess_bscan_crop_coords(mbscan)
    bscans_and_indices = [(a[z1:z2,left_crop:-right_crop],b) for a,b in bscans_and_indices]


    def save_bscan(tup):
        bscan = tup[0]
        fn = tup[1][0]
        s = tup[1][1]
        v = tup[1][2]
        if n_vol==1:
            outfn = os.path.join(bscan_folder,'complex_%05d.npy'%s)
        elif n_vol>1:
            volume_folder = os.path.join(bscan_folder,'volume_%02d'%v)
            os.makedirs(volume_folder,exist_ok=True)
            outfn = os.path.join(volume_folder,'complex_%05d.npy'%s)
        np.save(outfn,bscan)
        print('Saving volume %d bscan %d to %s'%(v,s,outfn))
        return 1

    try:
        p = mp.Pool(n_cpus)
        success = p.map(save_bscan,bscans_and_indices)
    except Exception as e:
        print(e)
        if require_multiprocessing:
            sys.exit('Multiprocessing failed. Serial processing aborted because require_multiprocessing=True.')
        success = []
        for bscan,(fn,s,v) in bscans_and_indices:
            if n_vol==1:
                outfn = os.path.join(bscan_folder,'complex_%05d.npy'%s)
            elif n_vol>1:
                volume_folder = os.path.join(bscan_folder,'volume_%02d'%v)
                os.makedirs(volume_folder,exist_ok=True)
                outfn = os.path.join(volume_folder,'complex_%05d.npy'%s)

            # if show_bscans:
            #     plt.cla()
            #     plt.imshow(blobf.dB(bscan),cmap='gray',clim=dB_clims)
            #     plt.title('volume %d, scan %d'%(v,s))
            #     png_fn = os.path.join(png_folder,'amplitude_%05d.png'%s)
            #     plt.savefig(png_fn)
            #     plt.pause(0.00001)

            np.save(outfn,bscan)
            print('Saving volume %d bscan %d to %s'%(v,s,outfn))
            success.append(True)

    if show_bscans or save_bscan_pngs:
        for bscan,(fn,s,v) in bscans_and_indices:
            plt.cla()
            plt.imshow(blobf.dB(bscan),cmap='gray',clim=dB_clims)
            plt.title('%s, volume %d, scan %d'%(fn,v,s))
            png_fn = os.path.join(png_folder,'amplitude_%05d.png'%s)
            plt.savefig(png_fn)
            if show_bscans:
                plt.pause(0.00000001)
                
        plt.close('all')
        sys.exit()
