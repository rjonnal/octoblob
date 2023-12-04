import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os,glob
import pathlib
import scipy.signal as sps
import multiprocessing as mp
from config import require_multiprocessing
import json

max_n_files = 600

def load_dict(fn):
    with open(fn,'r') as fid:
        s = fid.read()
        d = json.loads(s)
    return d

def save_dict(fn,d):
    s = json.dumps(d)
    with open(fn,'w') as fid:
        fid.write(s)
        
def xcorr(tup):
    ref = tup[0]
    tar = tup[1]
    nxc = np.abs(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))
    pidx = np.argmax(nxc)
    p = np.max(nxc)
    if pidx>len(tar)//2:
        pidx = pidx-len(tar)
    return p,pidx


def dB(arr):
    return 20*np.log10(arr)


def estimate_params(vol,image_type,border=20):
    print('Estimating parameters.')
    prof = np.mean(np.mean(vol,axis=2),axis=0)

    if False:
        noise_floor = sorted(prof)[:50]
        noise_floor = np.mean(noise_floor)
        pmax = np.max(prof)
        ithresh = noise_floor + 0.05*(pmax-noise_floor)
        othresh = noise_floor + 0.2*(pmax-noise_floor)

        ivalid = np.where(prof>ithresh)[0]
        v1 = ivalid[0]
        ovalid = np.where(prof>othresh)[0]
        v2 = ovalid[-1]

        z1 = v1-border
        z2 = v2+border
    else:
        pmax_index = np.argmax(prof)
        z1 = pmax_index-240
        z2 = pmax_index+120
        
    z2 = min(len(prof),z2)
    z1 = max(0,z1)

    
    sy,sz,sx = vol.shape
    x1 = 0
    x2 = sx-25
    y1 = 0
    y2 = sy
    smin = np.min(vol)
    smax = np.max(vol)

    if image_type=='dB':
        cmin,cmax = np.percentile(vol,(10,99.5))
    else:
        cmin = smin
        cmax = smax
        
    z1 = int(z1)
    z2 = int(z2)
    cmin = float(cmin)
    cmax = float(cmax)
    
    params =  {'x1':x1,
               'x2':x2,
               'y1':y1,
               'y2':y2,
               'z1':z1,
               'z2':z2,
               'mx':3,
               'my':3,
               'mz':3,
               'cmin':cmin,
               'cmax':cmax}

    return params
    

def valid_data(folder):

    print('Checking validity of %s.'%folder)
    
    # check that the .npy files in this folder form a valid OCT series/volume
    flist = sorted(glob.glob(os.path.join(folder,'*.npy')))

    if len(flist)<50:
        print('Invalid: fewer than 50 files.')
        return False

    test = np.load(flist[0])
    sz,sx = test.shape

    if sz<100:
        print('Invalid: depth less than 100.')
        return False
    
    if sx<100:
        print('Invalid: width less than 100.')
        return False
    
    return True

def get_flatten_info(source_volume,x=False,serial_flattening=False):
    print('Getting flattening info.')
    
    if x:
        source_volume = np.transpose(source_volume,[2,1,0])
        
    n_candidates = 10
    ref_candidate_indices = range(0,source_volume.shape[0],source_volume.shape[0]//n_candidates)

    ref_candidate_bscans = []
    for rci in ref_candidate_indices:
        ref_candidate_bscans.append(np.abs(source_volume[rci,:,:]))

    ref_candidates = [np.mean(b,axis=1) for b in ref_candidate_bscans]

    xcmat = np.ones((n_candidates,n_candidates))*np.nan

    for ridx1 in range(n_candidates):
        for ridx2 in range(ridx1,n_candidates):
            p, pidx = xcorr((ref_candidates[ridx1],ref_candidates[ridx2]))
            xcmat[ridx1,ridx2] = p
            xcmat[ridx2,ridx1] = p

    winner = np.argmax(np.sum(xcmat,axis=0))
    ref = ref_candidates[winner]
    ref_index = ref_candidate_indices[winner]

    tars = [np.mean(np.abs(b),axis=1) for b in source_volume]
    refs = [ref]*len(tars)
    tups = list(zip(tars,refs))
    try:
        assert serial_flattening==False
        n_cpus = os.cpu_count()
        p = mp.Pool(n_cpus)
        xcorr_output = p.map(xcorr,tups)
    except Exception as e:
        print(e)
        xcorr_output = []
        for tup in tups:
            xcorr_output.append(xcorr(tup))

    corrs,shifts = zip(*xcorr_output)
    if x:
        source_volume = np.transpose(source_volume,[2,1,0])
    return np.array(corrs),np.array(shifts,dtype=int)


def flatten_to(source_volume,shifts,x=False):
    print('Flattening to...')
    if x:
        source_volume = np.transpose(source_volume,[2,1,0])
    
    for y in range(source_volume.shape[0]):
        source_volume[y,:,:] = np.roll(source_volume[y,:,:],shifts[y],axis=0)
        
    if x:
        source_volume = np.transpose(source_volume,[2,1,0])
        
    return source_volume


def flatten(source_volume,thresh=0.5,x=False):
    corrs,shifts = get_flatten_info(source_volume,x=x)
    shifts[np.where(corrs<thresh*np.max(corrs))] = 0
    source_volume = flatten_to(source_volume,shifts,x=x)
    return source_volume


def process_amp(src,dest,redo=False,show_plots=True,serial_flattening=False):
    print('Running process_amp on %s.'%dest)
    if not valid_data(src):
        return
    
    enface_folder = os.path.join(dest,'enface')
    bscan_folder = os.path.join(dest,'bscan')

    os.makedirs(enface_folder,exist_ok=True)
    os.makedirs(bscan_folder,exist_ok=True)
    
    n_existing_enface = len(glob.glob(os.path.join(enface_folder,'*.png')))
    n_existing_bscan = len(glob.glob(os.path.join(bscan_folder,'*.png')))

    if n_existing_enface>50 and n_existing_bscan>50 and not redo:
        print('%d bscan and %d enface pngs exist; re-run with redo=True'%(n_existing_bscan,n_existing_enface))
        return
    
    
    n_existing_enface = len(glob.glob(os.path.join(enface_folder,'*.png')))
    n_existing_bscan = len(glob.glob(os.path.join(bscan_folder,'*.png')))

    if n_existing_enface>50 and n_existing_bscan>50 and not redo:
        print('%d bscan and %d enface pngs exist; re-run with redo=True'%(n_existing_bscan,n_existing_enface))
        return
    
    flist = glob.glob(os.path.join(src,'*.npy'))
    flist.sort()

    
    print('Loading volume from %s.'%src)
    vol = np.array([np.abs(np.load(f)) for f in flist])
    flatten_fn = os.path.join(src,'flatten.txt')
    try:
        shifts = np.loadtxt(flatten_fn)
        shifts = shifts.astype(int)
        print('Loaded y flattening data from %s.'%flatten_fn)
    except:
        print('Flattening...')
        corrs,shifts = get_flatten_info(vol,serial_flattening=serial_flattening)
        np.savetxt(flatten_fn,shifts)
        print('Saved y flatten data to %s.'%flatten_fn)
        
    vol = flatten_to(vol,shifts)

    flatten_x_fn = os.path.join(src,'flatten_x.txt')
    try:
        x_shifts = np.loadtxt(flatten_x_fn)
        x_shifts = x_shifts.astype(int)
        print('Loaded x flattening data from %s.'%flatten_x_fn)
    except:
        print('Flattening...')
        corrs,x_shifts = get_flatten_info(vol,x=True,serial_flattening=serial_flattening)
        np.savetxt(flatten_x_fn,x_shifts)
        print('Saved x flatten data to %s.'%flatten_x_fn)
        
    vol = flatten_to(vol,x_shifts,x=True)
    
    vol = dB(vol)

    params_fn = os.path.join(src,'visualization_parameters.json')
    try:
        params = load_dict(params_fn)
        assert len(params.keys())>0
        print('Loaded %s from %s.'%(params,params_fn))
    except:
        params = estimate_params(vol,'dB')
        save_dict(params_fn,params)

    display_volume = np.zeros((params['y2']-params['y1'],params['z2']-params['z1'],params['x2']-params['x1']))
    display_volume[:,:,:] = vol[params['y1']:params['y2'],params['z1']:params['z2'],params['x1']:params['x2']]
    kernel = np.ones((params['my'],params['mz'],params['mx']))
    display_volume = sps.fftconvolve(display_volume,kernel)/np.sum(kernel)
    display_volume = np.clip(display_volume,params['cmin'],params['cmax'])

    sy,sz,sx = display_volume.shape

    screen_dpi = 100
    print_dpi = 100

    do_enface = True
    do_bscans = True
    do_bscans_full = False
    
    if do_enface:
        fig = plt.figure(figsize=(sx*2/screen_dpi,max(sy,sz)/screen_dpi))
        rect = fig.patch
        rect.set_facecolor('black')

        ax1 = fig.add_axes([0,0,0.5,1.0])
        ax2 = fig.add_axes([0.5,(sy-sz)/sy,0.5,sz/sy])
        ax1.set_xticks([])
        ax2.set_xticks([])
        disp_bscan = np.mean(display_volume[sy//2-3:sy//2+3,:,:],axis=0)
        label = src.split('/')[1]

        for z in range(0,display_volume.shape[1]):
            ax1.clear()
            ax1.imshow(display_volume[:,z,:],clim=(params['cmin'],params['cmax']),cmap='gray')
            ax2.clear()
            ax2.imshow(disp_bscan,clim=(params['cmin'],params['cmax']),cmap='gray')
            ax2.axhline(z)
            ax2.text(sx-5,sy-5,label,color='white',ha='right',va='bottom')
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])
            
            outfn = os.path.join(enface_folder,'enface_%05d.png'%z)
            print(outfn)
            fig.savefig(outfn,facecolor=fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
            if show_plots:
                plt.pause(.000001)
        plt.close()

    if do_bscans_full:

        fig = plt.figure(figsize=(sx*2/screen_dpi,max(sy,sz)/screen_dpi))
        rect = fig.patch
        rect.set_facecolor('black')

        ax1 = fig.add_axes([0.0,(sy-sz)/sy,0.5,sz/sy])
        ax2 = fig.add_axes([0.5,0,0.5,1.0])
        ax1.set_xticks([])
        ax2.set_xticks([])
        disp_enface = np.max(display_volume,axis=1)
        label = src.split('/')[1]
        for y in range(0,display_volume.shape[0]):
            ax1.clear()
            ax1.imshow(display_volume[y,:,:],clim=(params['cmin'],params['cmax']),cmap='gray')
            
            ax2.clear()
            #ax2.imshow(disp_enface,clim=(params['cmin'],params['cmax']),cmap='gray')
            ax2.imshow(disp_enface,cmap='gray')
            ax2.axhline(y)
            #ax1.text(sx-5,sy-5,label,color='white',ha='right',va='bottom')
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])

            outfn = os.path.join(bscan_folder,'bscan_%05d.png'%y)
            print(outfn)
            fig.savefig(outfn,facecolor=fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
            if show_plots:
                plt.pause(.000001)
        plt.close()

    if do_bscans:

        fig = plt.figure(figsize=(sx/screen_dpi,sz/screen_dpi))
        rect = fig.patch
        rect.set_facecolor('black')

        ax1 = fig.add_axes([0.0,0.0,1.0,1.0])
        label = src.split('/')[1]
        for y in range(0,display_volume.shape[0]):
            ax1.clear()
            ax1.imshow(display_volume[y,:,:],clim=(params['cmin'],params['cmax']),cmap='gray')
            ax1.text(sx-5,5,label,color='white',ha='right',va='top')
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax1.set_yticks([])
            ax2.set_yticks([])

            outfn = os.path.join(bscan_folder,'bscan_%05d.png'%y)
            print(outfn)
            fig.savefig(outfn,facecolor=fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
            if show_plots:
                plt.pause(.000001)
        plt.close()


def process_octa(src,dest,redo=False,show_plots=True,serial_flattening=False):
    print('Running process_octa on %s.'%dest)

    pv_enface_folder = os.path.join(dest,'pv_enface')
    pv_bscan_folder = os.path.join(dest,'pv_bscan')
    
    av_enface_folder = os.path.join(dest,'av_enface')
    av_bscan_folder = os.path.join(dest,'av_bscan')

    os.makedirs(pv_enface_folder,exist_ok=True)
    os.makedirs(pv_bscan_folder,exist_ok=True)
    os.makedirs(av_enface_folder,exist_ok=True)
    os.makedirs(av_bscan_folder,exist_ok=True)
    
    n_existing_enface = len(glob.glob(os.path.join(av_enface_folder,'*.png')))
    n_existing_bscan = len(glob.glob(os.path.join(av_bscan_folder,'*.png')))

    if n_existing_enface>50 and n_existing_bscan>50 and not redo:
        print('%d bscan and %d enface pngs exist; re-run with redo=True'%(n_existing_bscan,n_existing_enface))
        return


    amp_src = os.path.join(src,'amplitude')
    av_src = os.path.join(src,'amplitude_variance')
    pv_src = os.path.join(src,'phase_variance')

    if not valid_data(amp_src):
        return
    
    if not valid_data(av_src):
        return
    
    if not valid_data(pv_src):
        return

    
    amp_flist = sorted(glob.glob(os.path.join(amp_src,'*.npy')))
    av_flist = sorted(glob.glob(os.path.join(av_src,'*.npy')))
    pv_flist = sorted(glob.glob(os.path.join(pv_src,'*.npy')))

    print('Loading volume from %s.'%amp_src)
    amp_vol = np.array([np.abs(np.load(f)) for f in amp_flist])
    
    flatten_fn = os.path.join(amp_src,'flatten.txt')
    try:
        shifts = np.loadtxt(flatten_fn)
        shifts = shifts.astype(int)
        print('Loaded y flattening data from %s.'%flatten_fn)
    except:
        print('Flattening...')
        corrs,shifts = get_flatten_info(amp_vol,serial_flattening=serial_flattening)
        np.savetxt(flatten_fn,shifts)
        print('Saved y flatten data to %s.'%flatten_fn)


    amp_vol = flatten_to(amp_vol,shifts)

    flatten_x_fn = os.path.join(amp_src,'flatten_x.txt')
    try:
        x_shifts = np.loadtxt(flatten_x_fn)
        x_shifts = x_shifts.astype(int)
        print('Loaded x flattening data from %s.'%flatten_x_fn)
    except:
        print('Flattening...')
        xcorrs,x_shifts = get_flatten_info(amp_vol,x=True,serial_flattening=serial_flattening)
        np.savetxt(flatten_x_fn,x_shifts)
        print('Saved x flatten data to %s.'%flatten_x_fn)

    amp_vol = flatten_to(amp_vol,x_shifts,x=True)
    amp_vol = dB(amp_vol)


    amp_params_fn = os.path.join(amp_src,'visualization_parameters.json')
    av_params_fn = os.path.join(av_src,'visualization_parameters.json')
    pv_params_fn = os.path.join(pv_src,'visualization_parameters.json')
    try:
        amp_params = load_dict(amp_params_fn)
        assert len(amp_params.keys())>0
        print('Loaded %s from %s.'%(amp_params,amp_params_fn))
    except:
        amp_params = estimate_params(amp_vol,'dB')
        save_dict(amp_params_fn,amp_params)

        
    amin = np.min(amp_vol)
    amax = np.max(amp_vol)
    thresh = amin + (amax-amin)*0.6

    print('Loading volume from %s.'%av_src)
    av_vol = np.array([np.abs(np.load(f)) for f in av_flist])
    print('Loading volume from %s.'%pv_src)
    pv_vol = np.array([np.abs(np.load(f)) for f in pv_flist])
    av_vol = flatten_to(av_vol,shifts)
    pv_vol = flatten_to(pv_vol,shifts)

    av_vol = flatten_to(av_vol,x_shifts,x=True)
    pv_vol = flatten_to(pv_vol,x_shifts,x=True)


    
    try:
        av_params = load_dict(av_params_fn)
        assert len(av_params.keys())>0
        print('Loaded %s from %s.'%(av_params,av_params_fn))
    except:
        av_params = dict(amp_params)
        av_params['cmin'] = 0.0
        av_params['cmax'] = np.max(av_vol)*0.0005
        save_dict(av_params_fn,av_params)

        
    try:
        pv_params = load_dict(pv_params_fn)
        assert len(pv_params.keys())>0
        print('Loaded %s from %s.'%(pv_params,pv_params_fn))
    except:
        pv_params = dict(amp_params)
        pv_params['cmin'] = 0.0
        pv_params['cmax'] = np.max(pv_vol)*0.01
        save_dict(pv_params_fn,pv_params)
    
    mask = np.zeros(amp_vol.shape,dtype=bool)
    mask[np.where(amp_vol>thresh)] = 1

    maskprof = np.mean(np.mean(mask.astype(float),axis=2),axis=0)
    
    proj1 = np.where(maskprof>0)[0][0]
    proj2 = proj1+20

    if False:
        plt.plot(maskprof)
        plt.axvline(proj1)
        plt.axvline(proj2)
        plt.show()
        sys.exit()
    
    av_vol = av_vol*mask
    pv_vol = pv_vol*mask
    
    amp_display_volume = np.zeros((amp_params['y2']-amp_params['y1'],amp_params['z2']-amp_params['z1'],amp_params['x2']-amp_params['x1']))
    amp_display_volume[:,:,:] = amp_vol[amp_params['y1']:amp_params['y2'],amp_params['z1']:amp_params['z2'],amp_params['x1']:amp_params['x2']]
    kernel = np.ones((amp_params['my'],amp_params['mz'],amp_params['mx']))
    amp_display_volume = sps.fftconvolve(amp_display_volume,kernel)/np.sum(kernel)
    amp_display_volume = np.clip(amp_display_volume,amp_params['cmin'],amp_params['cmax'])
    del amp_vol
    
    av_display_volume = np.zeros((av_params['y2']-av_params['y1'],av_params['z2']-av_params['z1'],av_params['x2']-av_params['x1']))
    av_display_volume[:,:,:] = av_vol[av_params['y1']:av_params['y2'],av_params['z1']:av_params['z2'],av_params['x1']:av_params['x2']]
    kernel = np.ones((av_params['my'],av_params['mz'],av_params['mx']))
    av_display_volume = sps.fftconvolve(av_display_volume,kernel)/np.sum(kernel)
    av_display_volume = np.clip(av_display_volume,av_params['cmin'],av_params['cmax'])
    del av_vol
    
    pv_display_volume = np.zeros((pv_params['y2']-pv_params['y1'],pv_params['z2']-pv_params['z1'],pv_params['x2']-pv_params['x1']))
    pv_display_volume[:,:,:] = pv_vol[pv_params['y1']:pv_params['y2'],pv_params['z1']:pv_params['z2'],pv_params['x1']:pv_params['x2']]
    kernel = np.ones((pv_params['my'],pv_params['mz'],pv_params['mx']))
    pv_display_volume = sps.fftconvolve(pv_display_volume,kernel)/np.sum(kernel)
    pv_display_volume = np.clip(pv_display_volume,pv_params['cmin'],pv_params['cmax'])
    del pv_vol

    sy,sz,sx = amp_display_volume.shape

    screen_dpi = 100
    print_dpi = 100

    do_enface = True
    do_bscans_full = False
    do_bscans = True
    do_av = True
    do_pv = False
    write_pngs = True
    
    if do_enface:
        v_fig = plt.figure(figsize=(sx*2/screen_dpi,(sy+sz)/screen_dpi))
        rect = v_fig.patch
        rect.set_facecolor('black')

        bscan_frac = sz/(sz+sy)
        enface_frac = sy/(sz+sy)
        
        amp_enface_ax = v_fig.add_axes([0.0,bscan_frac,0.5,enface_frac])
        v_enface_ax = v_fig.add_axes([0.5,bscan_frac,0.5,enface_frac])
        amp_bscan_ax = v_fig.add_axes([0.0,0.0,0.5,bscan_frac])
        v_bscan_ax = v_fig.add_axes([0.5,0.0,0.5,bscan_frac])
        axes = [amp_enface_ax,v_enface_ax,amp_bscan_ax,v_bscan_ax]
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        amp_disp_bscan = np.mean(amp_display_volume[sy//2-3:sy//2+3,:,:],axis=0)
        pv_disp_bscan = np.mean(pv_display_volume[sy//2-3:sy//2+3,:,:],axis=0)
        av_disp_bscan = np.mean(av_display_volume[sy//2-3:sy//2+3,:,:],axis=0)
        label = src.split('/')[1]

        if do_av:
            for z in range(0,amp_display_volume.shape[1]):
                for ax in axes:
                    ax.clear()
                    ax.set_xticks([])
                    ax.set_yticks([])
                amp_enface_ax.imshow(amp_display_volume[:,z,:],clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_enface_ax.imshow(av_display_volume[:,z,:],clim=(av_params['cmin'],av_params['cmax']),cmap='jet')
                amp_bscan_ax.imshow(amp_disp_bscan,clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_bscan_ax.imshow(av_disp_bscan,clim=(av_params['cmin'],av_params['cmax']),cmap='jet')

                amp_bscan_ax.axhline(z)
                v_bscan_ax.axhline(z)
                amp_bscan_ax.text(5,sy-5,label,color='white',ha='right',va='bottom')
                outfn = os.path.join(av_enface_folder,'enface_%05d.png'%z)
                print(outfn)
                if write_pngs:
                    v_fig.savefig(outfn,facecolor=v_fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
                if show_plots:
                    plt.pause(.000001)
                
        if do_pv:
            for z in range(0,amp_display_volume.shape[1]):
                for ax in axes:
                    ax.clear()
                    ax.set_xticks([])
                    ax.set_yticks([])
                amp_enface_ax.imshow(amp_display_volume[:,z,:],clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_enface_ax.imshow(pv_display_volume[:,z,:],clim=(pv_params['cmin'],pv_params['cmax']),cmap='jet')
                amp_bscan_ax.imshow(amp_disp_bscan,clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_bscan_ax.imshow(pv_disp_bscan,clim=(pv_params['cmin'],pv_params['cmax']),cmap='jet')

                amp_bscan_ax.axhline(z)
                v_bscan_ax.axhline(z)
                amp_bscan_ax.text(5,sy-5,label,color='white',ha='right',va='bottom')
                outfn = os.path.join(pv_enface_folder,'enface_%05d.png'%z)
                print(outfn)
                if write_pngs:
                    v_fig.savefig(outfn,facecolor=v_fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
                if show_plots:
                    plt.pause(.000001)
                
        plt.close()


    if do_bscans_full:
        v_fig = plt.figure(figsize=(sx*2/screen_dpi,(sy+sz)/screen_dpi))
        rect = v_fig.patch
        rect.set_facecolor('black')

        bscan_frac = sz/(sz+sy)
        enface_frac = sy/(sz+sy)
        
        amp_bscan_ax = v_fig.add_axes([0.0,enface_frac,0.5,bscan_frac])
        v_bscan_ax = v_fig.add_axes([0.5,enface_frac,0.5,bscan_frac])
        amp_enface_ax = v_fig.add_axes([0.0,0.0,0.5,enface_frac])
        v_enface_ax = v_fig.add_axes([0.5,0.0,0.5,enface_frac])
        axes = [amp_enface_ax,v_enface_ax,amp_bscan_ax,v_bscan_ax]
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        amp_disp_enface = np.max(amp_display_volume[:,proj1:proj2,:],axis=1)
        av_disp_enface = np.mean(av_display_volume[:,proj1:proj2,:],axis=1)
        pv_disp_enface = np.mean(pv_display_volume[:,proj1:proj2,:],axis=1)
        label = src.split('/')[1]

        if do_av:
            for y in range(0,amp_display_volume.shape[0]):
                for ax in axes:
                    ax.clear()
                amp_bscan_ax.imshow(amp_display_volume[y,:,:],clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_bscan_ax.imshow(av_display_volume[y,:,:],clim=(av_params['cmin'],av_params['cmax']),cmap='jet')
                amp_enface_ax.imshow(amp_disp_enface,cmap='gray')#,clim=(amp_params['cmin'],amp_params['cmax']))
                v_enface_ax.imshow(av_disp_enface,cmap='jet')#,clim=(av_params['cmin'],av_params['cmax']))

                amp_enface_ax.axhline(y)
                v_enface_ax.axhline(y)
                amp_enface_ax.text(5,sy-5,label,color='white',ha='right',va='bottom')
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                outfn = os.path.join(av_enface_folder,'enface_%05d.png'%y)
                print(outfn)
                if write_pngs:
                    v_fig.savefig(outfn,facecolor=v_fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
                if show_plots:
                    plt.pause(.000001)
                
        if do_pv:
            for y in range(0,amp_display_volume.shape[0]):
                for ax in axes:
                    ax.clear()
                amp_bscan_ax.imshow(amp_display_volume[y,:,:],clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_bscan_ax.imshow(pv_display_volume[y,:,:],clim=(pv_params['cmin'],pv_params['cmax']),cmap='jet')
                amp_enface_ax.imshow(amp_disp_enface,cmap='gray')#,clim=(amp_params['cmin'],amp_params['cmax']))
                v_enface_ax.imshow(pv_disp_enface,cmap='jet')#,clim=(pv_params['cmin'],pv_params['cmax']))

                amp_enface_ax.axhline(y)
                v_enface_ax.axhline(y)
                amp_enface_ax.text(5,sy-5,label,color='white',ha='right',va='bottom')
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                outfn = os.path.join(pv_enface_folder,'enface_%05d.png'%y)
                print(outfn)
                if write_pngs:
                    v_fig.savefig(outfn,facecolor=v_fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
                if show_plots:
                    plt.pause(.000001)
        plt.close()
                
    if do_bscans:
        v_fig = plt.figure(figsize=(sx*2/screen_dpi,sz/screen_dpi))
        rect = v_fig.patch
        rect.set_facecolor('black')

        amp_bscan_ax = v_fig.add_axes([0.0,0.0,0.5,1.0])
        v_bscan_ax = v_fig.add_axes([0.5,0.0,0.5,1.0])
        axes = [amp_bscan_ax,v_bscan_ax]
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        label = src.split('/')[1]

        if do_av:
            for y in range(0,amp_display_volume.shape[0]):
                for ax in axes:
                    ax.clear()
                amp_bscan_ax.imshow(amp_display_volume[y,:,:],clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_bscan_ax.imshow(av_display_volume[y,:,:],clim=(av_params['cmin'],av_params['cmax']),cmap='jet')

                amp_bscan_ax.text(5,sz-5,label,color='white',ha='right',va='bottom')
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                outfn = os.path.join(av_bscan_folder,'av_bscan_%05d.png'%y)
                print(outfn)
                if write_pngs:
                    v_fig.savefig(outfn,facecolor=v_fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
                if show_plots:
                    plt.pause(.000001)
                
        if do_pv:
            for y in range(0,amp_display_volume.shape[0]):
                for ax in axes:
                    ax.clear()
                amp_bscan_ax.imshow(amp_display_volume[y,:,:],clim=(amp_params['cmin'],amp_params['cmax']),cmap='gray')
                v_bscan_ax.imshow(pv_display_volume[y,:,:],clim=(pv_params['cmin'],pv_params['cmax']),cmap='jet')

                amp_bscan_ax.text(5,sz-5,label,color='white',ha='right',va='bottom')
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                outfn = os.path.join(pv_bscan_folder,'pv_bscan_%05d.png'%y)
                print(outfn)
                if write_pngs:
                    v_fig.savefig(outfn,facecolor=v_fig.get_facecolor(),edgecolor='none',dpi=print_dpi)
                if show_plots:
                    plt.pause(.000001)
                
        plt.close()

def process_amp_mp(tup):
    src = tup[0]
    dest = tup[1]
    process_amp(src,dest,redo=True,show_plots=False,serial_flattening=True)

def process_octa_mp(tup):
    src = tup[0]
    dest = tup[1]
    process_octa(src,dest,redo=True,show_plots=False,serial_flattening=True)
    


if __name__=='__main__':

    root = sys.argv[1]
    root_list = root.split('/')
    root_list[0] = root_list[0]+'_viz'
    dest_root = '/'.join(root_list)
    #dest_root = root+'_viz'
    p = pathlib.Path(root)
    all_folders = list(p.glob('**'))

    do_amp = 'amp' in sys.argv
    if do_amp:
        sys.argv.pop(sys.argv.index('amp'))
    do_octa = 'octa' in sys.argv
    if do_octa:
        sys.argv.pop(sys.argv.index('octa'))
    redo = 'redo' in sys.argv
    if redo:
        sys.argv.pop(sys.argv.index('redo'))
    parallel  = 'parallel' in sys.argv
    fast = 'fast' in sys.argv
    if fast:
        sys.argv.pop(sys.argv.index('fast'))
        
    if parallel:
        sys.argv.pop(sys.argv.index('parallel'))
        
    show_plots = (not parallel) and (not fast)
    

    if do_amp:
        amp_folders = [str(k) for k in all_folders if k.parts[-1][-7:]=='_bscans']
        amp_to_process = []
        for af in amp_folders:
            nfiles = len(glob.glob(os.path.join(af,'complex*.npy')))
            if nfiles<=max_n_files:
                bad_data_file = os.path.join(af,'BAD_DATA')
                if not os.path.exists(bad_data_file):
                    amp_to_process.append(af)
                else:
                    print('%s exists, skipping %s.'%(bad_data_file,af))

        if not parallel:
            for src in amp_to_process:
                dest = src.replace(root,dest_root)
                try:
                    process_amp(src,dest,redo=redo,show_plots=show_plots)
                except Exception as e:
                    print(e)
        else:
            n_cpus = 2#os.cpu_count()
            p = mp.Pool(n_cpus)
            dests = [src.replace(root,dest_root) for src in amp_to_process]
            tups = zip(amp_to_process,dests)
            #func = lambda tup: process_amp(tup[0],tup[1],redo=redo,show_plots=show_plots)
            amp_output = p.map(process_amp_mp,tups)

    if do_octa:
        octa_folders = [str(k) for k in all_folders if k.parts[-1]=='octa']
        octa_to_process = []
        for of in octa_folders:
            toks = [k for k in pathlib.Path(of).parts]
            toks = toks[:-1]+['BAD_DATA']
            bad_data_file = '/'.join(toks)
            if os.path.exists(bad_data_file):
                print('%s exists, skipping %s.'%(bad_data_file,of))
                continue
            
            n_amp_files = len(glob.glob(os.path.join(of,'amplitude/amp*.npy')))
            n_av_files = len(glob.glob(os.path.join(of,'amplitude_variance/av*.npy')))
            n_pv_files = len(glob.glob(os.path.join(of,'phase_variance/pv*.npy')))
            if n_amp_files==n_av_files==n_pv_files and n_amp_files>0:
                octa_to_process.append(of)

        if not parallel:
            for src in octa_to_process:
                dest = src.replace(root,dest_root)
                try:
                    process_octa(src,dest,redo=redo,show_plots=show_plots)
                except Exception as e:
                    print(e)
        else:
            n_cpus = 2#os.cpu_count()
            p = mp.Pool(n_cpus)
            dests = [src.replace(root,dest_root) for src in octa_to_process]
            tups = zip(octa_to_process,dests)
            #func = lambda tup: process_octa(tup[0],tup[1],redo=redo,show_plots=show_plots)
            octa_output = p.map(process_octa_mp,tups)
            
