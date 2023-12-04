from matplotlib import pyplot as plt
import numpy as np
import sys,os,glob,shutil
import pathlib
import multiprocessing as mp
from config import stimulus_index
from matplotlib.widgets import Slider, Button
import scipy.signal as sps

mp.set_start_method('fork')


org_ylim = (-200,200)

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

def get_flatten_info(source_volume,x=False,serial_flattening=False,medfilt=15):
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

    if not medfilt==1:
        shifts = sps.medfilt(shifts,medfilt)

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


if __name__=='__main__':

    root_folder = sys.argv[1]

    output_folder = os.path.join(root_folder,'org_data')
    os.makedirs(output_folder,exist_ok=True)

    flatten_fast = 'flatten_fast' in sys.argv
    flatten_slow = 'flatten_slow' in sys.argv
    
    amplitude_folder = os.path.join(root_folder,'block_amp')
    phase_velocity_folder = os.path.join(root_folder,'phase_velocity')
    complex_variance_folder = os.path.join(root_folder,'block_var')
    residual_error_folder = os.path.join(root_folder,'residual_error')
    
    amp_files = sorted(glob.glob(os.path.join(amplitude_folder,'*.npy')))
    pv_files = sorted(glob.glob(os.path.join(phase_velocity_folder,'*.npy')))
    cv_files = sorted(glob.glob(os.path.join(complex_variance_folder,'*.npy')))
    re_files = sorted(glob.glob(os.path.join(residual_error_folder,'*.npy')))

    amp_vol = np.array([np.load(f) for f in amp_files])
    pv_vol = np.array([np.load(f) for f in pv_files])
    cv_vol = np.array([np.load(f) for f in cv_files])
    re_vol = np.array([np.load(f) for f in re_files])


    prof = np.mean(np.mean(amp_vol,axis=2),axis=0)
    pmax = np.argmax(prof)
    z1,z2 = pmax-180,pmax+60
    
    z1 = max(0,z1)
    z2 = min(len(prof),z2)

    amp_vol = amp_vol[:,z1:z2,:]
    pv_vol = pv_vol[:,z1:z2,:]
    cv_vol = cv_vol[:,z1:z2,:]
    re_vol = re_vol[:,z1:z2,:]
    

    if flatten_slow:
        y_flatten_fn = os.path.join(root_folder,'flatten_y.txt')
        try:
            y_shifts = np.loadtxt(y_flatten_fn)
            y_shifts = y_shifts.astype(int)
            print('Loaded y flattening data from %s.'%y_flatten_fn)
        except Exception as e:
            corrs,y_shifts = get_flatten_info(amp_vol)
            y_shifts[np.where(corrs<np.max(corrs)*0.5)] = 0
            np.savetxt(y_flatten_fn,y_shifts)

        amp_vol = flatten_to(amp_vol,y_shifts)
        pv_vol = flatten_to(pv_vol,y_shifts)
        cv_vol = flatten_to(cv_vol,y_shifts)
        re_vol = flatten_to(re_vol,y_shifts)

    var_vol = np.abs(cv_vol)
    var_vec = np.mean(np.mean(var_vol,axis=2),axis=1)

    if flatten_fast:
        x_flatten_fn = os.path.join(root_folder,'flatten_x.txt')
        try:
            x_shifts = np.loadtxt(x_flatten_fn)
            x_shifts = x_shifts.astype(int)
            print('Loaded y flattening data from %s.'%x_flatten_fn)
        except Exception as e:
            corrs,x_shifts = get_flatten_info(amp_vol,x=True)
            x_shifts[np.where(corrs<np.max(corrs)*0.5)] = 0
            np.savetxt(x_flatten_fn,x_shifts)

        amp_vol = flatten_to(amp_vol,x_shifts,x=True)
        pv_vol = flatten_to(pv_vol,x_shifts,x=True)
        cv_vol = flatten_to(cv_vol,x_shifts,x=True)
        re_vol = flatten_to(re_vol,x_shifts,x=True)

    bscan = amp_vol[stimulus_index,:,:]
    top = bscan[:-2,:]
    middle = bscan[1:-1]
    bottom = bscan[2:]
    peaks = np.zeros(bscan.shape)
    peaks[1:-1,:] = (middle>top)*(middle>bottom)

    fig = plt.figure(figsize=(12,8))
    img_ax = fig.add_axes([0.05,0.3,0.4,0.6])
    img_ax.imshow(dB(bscan),cmap='gray')


    plot_ax = fig.add_axes([0.55,0.3,0.4,0.6])

    fbutton_ax = fig.add_axes([0.05,0.1,0.2,0.1])
    dbutton_ax = fig.add_axes([0.3,0.1,0.2,0.1])
    sbutton_ax = fig.add_axes([0.55,0.1,0.2,0.1])
    
    top_vec = []
    bottom_vec = []

    def correct(x,y,peaks):
        x_peakidx = np.where(peaks[:,x])[0]
        d = np.abs(y-x_peakidx)
        winner = x_peakidx[np.argmin(d)]
        return x,winner
    
    counter = 0
    last_x = 0
    tops_all = []
    bottoms_all = []
    x_tops = []
    y_tops = []
    x_bottoms = []
    y_bottoms = []
    org_all = []


    def save():

        coords = list(zip(x_tops,y_tops,x_bottoms,y_bottoms))
        all_dat = []
        for c,o in zip(coords,org_all):
            all_dat.append(list(c)+list(o))

        np.savetxt(os.path.join(output_folder,'org_data.txt'),all_dat)
        np.save(os.path.join(output_folder,'amplitude_bscan.npy'),bscan)
        np.save(os.path.join(output_folder,'amplitude_correlations.txt'),var_vec)
            
    def fill(tol=1):
        global counter,last_x,pv_vol,top_vec,bottom_vec,plot_ax,org_all
        global tops_all,bottoms_all,img_ax,x_tops,y_tops,x_bottoms,y_bottoms
        if x_tops[-1]>x_tops[-2]:
            step = 1
        else:
            step = -1
            
        x1 = x_tops[-2]
        x2 = x_tops[-1]
        x_locs = list(range(x1,x2+1,step))
        ytop1 = y_tops[-2]
        ytop2 = y_tops[-1]
        ybottom1 = y_bottoms[-2]
        ybottom2 = y_bottoms[-1]
        ytop_fit = np.round(np.polyval(np.polyfit([x1,x2],[ytop1,ytop2],1),x_locs)).astype(int)
        ybottom_fit = np.round(np.polyval(np.polyfit([x1,x2],[ybottom1,ybottom2],1),x_locs)).astype(int)

        for idx in range(1,len(x_locs)-2):
            x = x_locs[idx]
            ytopcand = ytop_fit[idx]
            ybottomcand = ybottom_fit[idx]

            topymax = -np.inf
            bottomymax = -np.inf

            topwinner = ytopcand
            bottomwinner = ybottomcand
            
            for offset in range(-tol,tol+1):
                top_pixel = dB(bscan)[ytopcand+offset,x]
                bottom_pixel = dB(bscan)[ybottomcand+offset,x]
                
                if top_pixel>topymax:
                    topymax = top_pixel
                    topwinner = ytopcand+offset

                if bottom_pixel>bottomymax:
                    bottomymax = bottom_pixel
                    bottomwinner = ybottomcand+offset
                    
            x_tops.append(x)
            x_bottoms.append(x)
            y_tops.append(topwinner)
            y_bottoms.append(bottomwinner)
            tops_all.append(pv_vol[:,topwinner,x])
            bottoms_all.append(pv_vol[:,bottomwinner,x])
            org_all = [top-bottom for bottom,top in zip(bottoms_all,tops_all)]

        update_img()
        update_plot()
        
    
    def update_img():
        global counter,last_x,pv_vol,top_vec,bottom_vec,plot_ax,org_all
        global tops_all,bottoms_all,img_ax,x_tops,y_tops,x_bottoms,y_bottoms
        img_ax.clear()
        img_ax.imshow(dB(bscan),cmap='gray')
        img_ax.plot(x_tops,y_tops,'go',markersize=3)
        img_ax.plot(x_bottoms,y_bottoms,'go',markersize=3)
        plt.pause(0.001)
        
    def update_plot():
        global counter,last_x,pv_vol,top_vec,bottom_vec,plot_ax,org_all
        global tops_all,bottoms_all,img_ax,x_tops,y_tops,x_bottoms,y_bottoms
        plot_ax.clear()
        for resp in org_all:
            plot_ax.plot(resp,color='k',alpha=0.02)
        org_mean = np.nanmean(org_all,axis=0)
        plot_ax.plot(org_mean)
        plot_ax.set_ylim(org_ylim)
        plt.pause(0.001)

    def delete_last():
        global counter,last_x,pv_vol,top_vec,bottom_vec,plot_ax,org_all
        global tops_all,bottoms_all,img_ax,x_tops,y_tops,x_bottoms,y_bottoms
        org_all = org_all[:-1]
        x_tops = x_tops[:-1]
        y_tops = y_tops[:-1]
        x_bottoms = x_bottoms[:-1]
        y_bottoms = y_bottoms[:-1]
        tops_all = tops_all[:-1]
        bottoms_all = bottoms_all[:-1]

        update_img()
        update_plot()

    def onpress(event):
        global counter,last_x,pv_vol,top_vec,bottom_vec,plot_ax,org_all
        global tops_all,bottoms_all,img_ax,x_tops,y_tops,x_bottoms,y_bottoms
        if event.key=='backspace':
            delete_last()
        if event.key=='g':
            fill()
        
    def onclick(event):
        global counter,last_x,pv_vol,top_vec,bottom_vec,plot_ax,org_all
        global tops_all,bottoms_all,img_ax,x_tops,y_tops,x_bottoms,y_bottoms
        if event.inaxes==img_ax:
            if counter%2==0:
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                x,y = correct(x,y,peaks)
                last_x = x
                top_vec.append((x,y))
            else:
                x = last_x
                y = int(round(event.ydata))
                x,y = correct(x,y,peaks)
                bottom_vec.append((x,y))
            counter+=1

            if counter%2==1:
                tops_all.append(pv_vol[:,y,x])
                x_tops.append(x)
                y_tops.append(y)
                update_img()
            else:
                bottoms_all.append(pv_vol[:,y,x])
                x_bottoms.append(x)
                y_bottoms.append(y)
                update_img()
                
            if counter%2==0:
                org_all = [top-bottom for bottom,top in zip(bottoms_all,tops_all)]
                update_plot()
                    

    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    pid = fig.canvas.mpl_connect('key_press_event',onpress)

    btn_fill = Button(fbutton_ax,'Fill (g)',hovercolor='0.975')
    ffunc = lambda x: fill()
    btn_fill.on_clicked(ffunc)

    btn_del = Button(dbutton_ax,'Delete (backspace)',hovercolor='0.975')
    dfunc = lambda x: delete_last()
    btn_del.on_clicked(dfunc)

    btn_save = Button(sbutton_ax,'Save (s)',hovercolor='0.975')
    sfunc = lambda x: save()
    btn_save.on_clicked(sfunc)
    

    
    plt.show()
