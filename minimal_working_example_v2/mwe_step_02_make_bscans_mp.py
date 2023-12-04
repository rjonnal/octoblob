import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os,glob
import functions as blobf
from config import autocrop,use_generic_mapping_dispersion_file,start_bscan,end_bscan,autocrop_stride
from config import dB_clims,show_bscans,left_crop,right_crop,save_bscan_pngs
from config import require_multiprocessing, oversample
import multiprocessing as mp
import pathlib
import errno
import os
import datetime

mp.set_start_method('fork')

if __name__=='__main__':

    n_cpus = os.cpu_count()
    
    try:
        filt = sys.argv[1]
    except:
        print('Please supply a file or folder name at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
        sys.exit()


    files = list(pathlib.Path(filt).rglob('*.unp'))
    files = [str(f) for f in files]
    if len(files)==0:
        files = [filt]


    success_log_filename = 'make_bscans_successes.log'
    failure_log_filename = 'make_bscans_failures.log'
    with open(success_log_filename,'a') as fid:
        fid.write('\n[%s] ++++++++++ Starting ++++++++++\n'%datetime.datetime.now())

    with open(failure_log_filename,'a') as fid:
        fid.write('\n[%s] ++++++++++ Starting ++++++++++\n'%datetime.datetime.now())

    
    for fn in files:
        try:
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
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '%s_mapping_dispersion_coefficients.txt'%tag)

            def save_bscan(tup):
                src = tup[0]
                s = tup[1]
                folder  = tup[2]
                dest = os.path.join(folder,'complex_%05d.npy'%s)
                print('Saving bscan %d to %s.'%(s,dest))
                spectra = blobf.get_frame(src,s)
                bscan = blobf.spectra_to_bscan(spectra,coefs,oversample=oversample)
                np.save(dest,bscan)
                return dest

            tups = []

            for v in range(n_vol):
                for s in range(n_slow):

                    if start_bscan is not None:
                        if s<start_bscan:
                            continue
                    if end_bscan is not None:
                        if s>=end_bscan:
                            continue
                    tups.append((fn,s,bscan_folder))

            try:
                p = mp.Pool(n_cpus)
                bscan_filenames = p.map(save_bscan,tups)
            except Exception as e:
                print(e)
                if require_multiprocessing:
                    sys.exit('Multiprocessing failed. Serial processing aborted because require_multiprocessing=True.')
                bscan_filenames = []
                for tup in tups:
                    bscan_filenames.append(save_bscan(tup))


            # load some B-scans and average for cropping

            #bscan_filenames = glob.glob(os.path.join(bscan_folder,'complex_*.npy'))
            bscan_filenames.sort()

            bscans = [np.abs(np.load(bfn)) for bfn in bscan_filenames[::autocrop_stride]]

            bscans = np.array(bscans)
            mbscan = np.mean(bscans,axis=0)
            z1,z2 = blobf.guess_bscan_crop_coords(mbscan)

            def crop_bscan(tup):
                print('Cropping %s from %d to %d.'%tup)
                bfn = tup[0]
                z1 = tup[1]
                z2 = tup[2]
                uncropped = np.load(bfn)
                cropped = uncropped[z1:z2,:]
                np.save(bfn,cropped)
                return 1

            tups = [(bfn,z1,z2) for bfn in bscan_filenames]
            try:
                p = mp.Pool(n_cpus)
                success = p.map(crop_bscan,tups)
            except Exception as e:
                print(e)
                if require_multiprocessing:
                    sys.exit('Multiprocessing failed. Serial processing aborted because require_multiprocessing=True.')
                success = []
                for tup in tups:
                    success.append(crop_bscan(tup))

            if show_bscans or save_bscan_pngs:
                for bfn in bscan_filenames:
                    bscan = np.load(bfn)
                    plt.cla()
                    plt.imshow(blobf.dB(bscan),cmap='gray',clim=dB_clims)
                    plt.title('%s'%bfn)
                    png_fn = os.path.join(png_folder,'amplitude_%05d.png'%s)
                    plt.savefig(png_fn)
                    if show_bscans:
                        plt.pause(0.00000001)

                plt.close('all')

            with open(success_log_filename,'a') as fid:
                fid.write('[%s] %s\n'%(datetime.datetime.now(),fn))

        except Exception as e:
            with open(failure_log_filename,'a') as fid:
                fid.write('[%s] %s: %s\n'%(datetime.datetime.now(),fn,e))
