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

class Volume:
    def __init__(self,source_volume,visualization_parameters_fn,im_ax,image_type,enface=False,default_params=None):

        try:
            assert image_type in ['av','pv','structure','org']
        except AssertionError as ae:
            print('Invalid image type %s.'%image_type)
        
        if enface:
            source_volume = np.transpose(source_volume,[1,0,2])
        self.source_volume = source_volume
        if image_type=='structure':
            self.source_volume = dB(self.source_volume)
        self.display_volume = np.zeros(source_volume.shape)
        self.display_volume[...] = source_volume[...]
        self.sy,self.sz,self.sx = self.source_volume.shape
        self.smax = np.max(self.source_volume)
        self.smin = np.min(self.source_volume)
        if do_dB:
            self.smin = 40
        self.visualization_parameters_fn = visualization_parameters_fn
        self.folder = os.path.split(self.visualization_parameters_fn)[0]
        self.im_ax = im_ax
        self.im_ax.set_title(self.folder)

        if default_params is None:
            default_params =  {'x1':0,
                            'x2':self.sx-25,
                            'y1':0,
                            'y2':self.sy,
                            'z1':0,
                            'z2':self.sz,
                            'mx':3,
                            'my':3,
                            'mz':3,
                            'cmin':self.smin,
                            'cmax':self.smax}

        try:
            with open(self.visualization_parameters_fn,'r') as fid:
                s = fid.read()
                self.params = json.loads(s)
                print('Loading %s from %s.'%(s,self.visualization_parameters_fn))
        except Exception as e:
            print(e)
            self.params = default_params


        self.didx = self.sy//2
        self.dmin = self.params['y1']
        self.dmax = self.params['y2']
        self.cmap = 'gray'
        self.update()

    def save_parameters(self,event):
        with open(self.visualization_parameters_fn,'w') as fid:
            s = json.dumps(self.params)
            fid.write(s)
            print('Saving %s to %s.'%(s,self.visualization_parameters_fn))
            plt.close()

    def update(self):
        self.display_volume = np.zeros((self.params['y2']-self.params['y1'],self.params['z2']-self.params['z1'],self.params['x2']-self.params['x1']))
        self.display_volume[:,:,:] = self.source_volume[self.params['y1']:self.params['y2'],self.params['z1']:self.params['z2'],self.params['x1']:self.params['x2']]
        kernel = np.ones((self.params['my'],self.params['mz'],self.params['mx']))
        self.display_volume = sps.fftconvolve(self.display_volume,kernel)/np.sum(kernel)
        self.display_volume = np.clip(self.display_volume,self.params['cmin'],self.params['cmax'])
        self.show()

    def crop_x1(self,event,val):
        self.params['x1'] = val
        self.update()

    def crop_x2(self,event,val):
        self.params['x2'] = val
        self.update()

    def crop_y1(self,event,val):
        self.params['y1'] = val
        self.update()

    def crop_y2(self,event,val):
        self.params['y2'] = val
        self.update()

    def crop_z1(self,event,val):
        self.params['z1'] = val
        self.update()

    def crop_z2(self,event,val):
        self.params['z2'] = val
        self.update()

    def avg_x(self,event,val):
        self.params['mx'] = val
        self.update()

    def avg_y(self,event,val):
        self.params['my'] = val
        self.update()

    def avg_z(self,event,val):
        self.params['mz'] = val
        self.update()

    def set_cmin(self,event,val):
        self.params['cmin'] = val
        self.update()

    def set_cmax(self,event,val):
        self.params['cmax'] = val
        self.update()

    def set_didx(self,event,val):
        didx = int(round(val))
        self.didx = didx
        self.show()

    def show(self):
        self.im_ax.clear()
        self.im_ax.imshow(self.display_volume[self.didx,:,:],cmap=self.cmap,clim=(self.params['cmin'],self.params['cmax']))
        self.im_ax.set_title(self.folder)
        
    def set_image_axis(self,ax):
        self.im_ax = ax
        self.im_ax.set_title(self.folder)




def estimate_params(vol,image_type,border=20):
    
    prof = np.mean(np.mean(vol,axis=2),axis=0)

    noise_floor = sorted(prof)[:50]
    noise_floor = np.mean(noise_floor)
    pmax = np.max(prof)
    thresh = noise_floor + 0.05*(pmax-noise_floor)
    valid = np.where(prof>thresh)[0]
    v1 = valid[0]
    v2 = valid[-1]
    z1 = v1-border
    z2 = v2+border
    z2 = min(len(prof),z2)
    z1 = max(0,z1)

    
    
    plt.figure()
    plt.plot(prof)
    plt.axhline(thresh)
    plt.axvline(z1)
    plt.axvline(z2)
    plt.show()
    sys.exit()
        
def viewer(files,enface=False):

    files.sort()

    test = files[0]
    folder,fn = os.path.split(test)
    toks = fn.split('_')
    if toks[0] in ['amp','complex']:
        image_type = 'structure'
    elif toks[0] in ['av']:
        image_type = 'av'
    elif toks[0] in ['pv']:
        image_type = 'pv'
    else:
        print('Unknown image type (amp, complex are structure; av is amplitude variance; pv is phase variance: %s.'%test)


    visualization_parameters_fn = os.path.join(folder,'visualization_parameters.json')
    
    source_volume = np.array([np.load(f) for f in files])
    source_volume = np.abs(source_volume)

    flatten = True

    if flatten:
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
            n_cpus = os.cpu_count()
            p = mp.Pool(n_cpus)
            xcorr_output = p.map(xcorr,tups)
        except Exception as e:
            print(e)
            if require_multiprocessing:
                sys.exit('Multiprocessing failed. Serial processing aborted because require_multiprocessing=True.')
            xcorr_output = []
            for tup in tups:
                xcorr_output.append(xcorr(tup))

        corrs,shifts = zip(*xcorr_output)
        for y in range(source_volume.shape[0]):
            source_volume[y,:,:] = np.roll(source_volume[y,:,:],shifts[y],axis=0)

    fig = plt.figure(figsize=(16,6))
    ax_img = fig.add_axes([.1,.1,.4,.8])

    params = estimate_params(source_volume,image_type)
    
    vol = Volume(source_volume,visualization_parameters_fn=visualization_parameters_fn,im_ax=ax_img,image_type=image_type,enface=enface)
    #vol.set_image_axis(ax_img)

    # available adjustments:
    # each entry is [func_name,min_value,max_value,default_value,valstep]
    adjustments = {
        'crop_x1':[vol.crop_x1,0,vol.sx,0,1],
        'crop_x2':[vol.crop_x2,0,vol.sx,vol.sx,1],
        'crop_y1':[vol.crop_y1,0,vol.sy,0,1],
        'crop_y2':[vol.crop_y2,0,vol.sy,vol.sy,1],
        'crop_z1':[vol.crop_z1,0,vol.sz,0,1],
        'crop_z2':[vol.crop_z2,0,vol.sz,vol.sz,1],
        'avg_x':[vol.avg_x,1,10,1,1],
        'avg_y':[vol.avg_y,1,10,1,1],
        'avg_z':[vol.avg_z,1,10,1,1],
        'cmax':[vol.set_cmax,vol.smin,vol.smax,vol.smax,None],
        'cmin':[vol.set_cmin,vol.smin,vol.smax,vol.smin,None],
        'didx':[vol.set_didx,0,vol.sy,1]
    }

    n_adjustments = len(list(adjustments.keys()))
    slider_ax_h = 1.0/15.0
    slider_bottoms = np.arange(1.0-slider_ax_h,0,-slider_ax_h)
    slider_axes = []
    sliders = []
    functions = []

    kidx = 0
    sb = slider_bottoms[kidx]
    ax_0 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_0.set_xticks([])
    ax_0.set_yticks([])
    vmin = 0
    vmax = vol.sx
    default = vol.params['x1']
    valstep = 1
    slider_0 = Slider(ax=ax_0,label='crop_x1',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_0 = lambda event: vol.crop_x1(event,slider_0.val)
    slider_0.on_changed(curried_0)

    kidx = 1
    sb = slider_bottoms[kidx]
    ax_1 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_1.set_xticks([])
    ax_1.set_yticks([])
    vmin = 0
    vmax = vol.sx
    default = vol.params['x2']
    valstep = 1
    slider_1 = Slider(ax=ax_1,label='crop_x2',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_1 = lambda event: vol.crop_x2(event,slider_1.val)
    slider_1.on_changed(curried_1)


    kidx = 2
    sb = slider_bottoms[kidx]
    ax_2 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_2.set_xticks([])
    ax_2.set_yticks([])
    vmin = 0
    vmax = vol.sy
    default = vol.params['y1']
    valstep = 1
    slider_2 = Slider(ax=ax_2,label='crop_y1',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_2 = lambda event: vol.crop_y1(event,slider_2.val)
    slider_2.on_changed(curried_2)

    kidx = 3
    sb = slider_bottoms[kidx]
    ax_3 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_3.set_xticks([])
    ax_3.set_yticks([])
    vmin = 0
    vmax = vol.sy
    default = vol.params['y2']
    valstep = 1
    slider_3 = Slider(ax=ax_3,label='crop_y2',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_3 = lambda event: vol.crop_y2(event,slider_3.val)
    slider_3.on_changed(curried_3)

    kidx = 4
    sb = slider_bottoms[kidx]
    ax_4 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_4.set_xticks([])
    ax_4.set_yticks([])
    vmin = 0
    vmax = vol.sz
    default = vol.params['z1']
    valstep = 1
    slider_4 = Slider(ax=ax_4,label='crop_z1',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_4 = lambda event: vol.crop_z1(event,slider_4.val)
    slider_4.on_changed(curried_4)

    kidx = 5
    sb = slider_bottoms[kidx]
    ax_5 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_5.set_xticks([])
    ax_5.set_yticks([])
    vmin = 0
    vmax = vol.sz
    default = vol.params['z2']
    valstep = 1
    slider_5 = Slider(ax=ax_5,label='crop_z2',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_5 = lambda event: vol.crop_z2(event,slider_5.val)
    slider_5.on_changed(curried_5)

    average_max = 10

    kidx = 6
    sb = slider_bottoms[kidx]
    ax_6 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_6.set_xticks([])
    ax_6.set_yticks([])
    vmin = 1
    vmax = average_max
    default = vol.params['mx']
    valstep = 1
    slider_6 = Slider(ax=ax_6,label='avg_x',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_6 = lambda event: vol.avg_x(event,slider_6.val)
    slider_6.on_changed(curried_6)

    kidx = 7
    sb = slider_bottoms[kidx]
    ax_7 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_7.set_xticks([])
    ax_7.set_yticks([])
    vmin = 1
    vmax = average_max
    default = vol.params['my']
    valstep = 1
    slider_7 = Slider(ax=ax_7,label='avg_y',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_7 = lambda event: vol.avg_y(event,slider_7.val)
    slider_7.on_changed(curried_7)

    kidx = 8
    sb = slider_bottoms[kidx]
    ax_8 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_8.set_xticks([])
    ax_8.set_yticks([])
    vmin = 1
    vmax = average_max
    default = vol.params['mz']
    valstep = 1
    slider_8 = Slider(ax=ax_8,label='avg_z',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_8 = lambda event: vol.avg_z(event,slider_8.val)
    slider_8.on_changed(curried_8)

    kidx = 9
    sb = slider_bottoms[kidx]
    ax_9 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_9.set_xticks([])
    ax_9.set_yticks([])
    vmin = vol.smin
    vmax = vol.smax
    default = vol.params['cmin']
    valstep = None
    slider_9 = Slider(ax=ax_9,label='cmin',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_9 = lambda event: vol.set_cmin(event,slider_9.val)
    slider_9.on_changed(curried_9)

    kidx = 10
    sb = slider_bottoms[kidx]
    ax_10 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_10.set_xticks([])
    ax_10.set_yticks([])
    vmin = vol.smin
    vmax = vol.smax
    default = vol.params['cmax']
    valstep = None
    slider_10 = Slider(ax=ax_10,label='cmax',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_10 = lambda event: vol.set_cmax(event,slider_10.val)
    slider_10.on_changed(curried_10)

    kidx = 11
    sb = slider_bottoms[kidx]
    ax_11 = fig.add_axes([0.6,sb,0.3,slider_ax_h])
    ax_11.set_xticks([])
    ax_11.set_yticks([])
    vmin = vol.dmin
    vmax = vol.dmax-1
    default = vol.didx
    valstep = 1
    slider_11 = Slider(ax=ax_11,label='didx',valmin=vmin,valmax=vmax,valstep=valstep,valinit=default)
    curried_11 = lambda event: vol.set_didx(event,slider_11.val)
    slider_11.on_changed(curried_11)


    ax_12 = fig.add_axes([0.8,0.01,0.1,0.05])
    ax_12.set_xticks([])
    ax_12.set_yticks([])
    btn_save_params = Button(ax_12,'Save Params', hovercolor='0.975')
    curried_12 = lambda event: vol.save_parameters(event)
    btn_save_params.on_clicked(curried_12)

    plt.show()
    


if __name__=='__main__':

    if 'enface' in sys.argv:
        enface=True
        sys.argv.pop(sys.argv.index('enface'))
    else:
        enface=False
    
    try:
        root = sys.argv[1]
        filters = sys.argv[2:]
        p = pathlib.Path(root)
        all_folders = list(p.glob('**'))
        file_lists_to_process = []

        for folder in all_folders:
            for f in filters:
                test = os.path.join(folder,f)
                file_list = glob.glob(test)
                if len(file_list)>0 and len(file_list)<=max_n_files:
                    file_lists_to_process.append(sorted(file_list))
    except Exception as e:
        print(e)
        print('Please supply a root folder name at the command line, followed by filters, i.e., python volumetric_viewer.py data filt1 filt2')
        sys.exit()


    for idx,file_list in enumerate(file_lists_to_process):
        print('Working on set %d of %d: %d files containing %s... %s.'%(idx+1,len(file_lists_to_process),len(file_list),file_list[0],file_list[-1]))
        viewer(file_list,enface=enface)
