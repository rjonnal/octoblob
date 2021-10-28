import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob,os,sys,shutil
import scipy.signal as sps
import octoblob.plotting_functions as opf
import warnings
import octoblob as blob
import webbrowser as wb


################################### Parameters ##########################################
# parameters for now, but turn these into command line arguments
figure_mode = 'paper' # 'paper' or 'presentation'
index_of_stimulus = 48 # expressed in terms of the number of files

dx_um = 3.0
dy_um = 3.0
dz_um = 2.5
dt_s = 0.0025

# layer thickness, used for axial integration of signal:
layer_thickness = 1

# approximate region of stimulation, in fast axis coordinates:
stimulated_region_start = 0
stimulated_region_end = 170

bscan_vel_clim = (-4,4)
mscan_vel_clim = (-3,3)
abs_plot_clim = (-3.5,3.5)
rel_plot_clim = (-5,4)

# the time range, in ms, to include in plots
# stimulus is at t=0.0
tlim_ms = (-100,100)

# crop b-scans, just for visualization
bscan_crop_left = 10
bscan_crop_right = 20

vel_lateral_smoothing_sigma = 3 # use 0 for no smoothing

profile_peak_threshold = 5000
peak_labels = ['ISOS','COST']
layer_differences = [('COST','ISOS')]#,('RPE','ISOS')]

flatten = True # flatten B-scans using 1D A-scan registration
flattening_averaging_width = 5 # number of A-scans to average for flattening

profile_figure_size = (4,4)
plots_figure_size = (4,4)

bscan_figure_width = 4

screen_dpi = 100
print_dpi = 300

stim_color = 'g'
stim_linestyle = '-'
output_folder = 'org_block_figures'
auto_open_report = True
make_pdf = False # requires pandoc
################################### End parameters ######################################

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

#folders = glob.glob('*_bscans/cropped/phase_ramps_007ms_npy')
args = sys.argv[1:]
folders = []

args = blob.expand_wildcard_arguments(args)

def usage():
    print('Usage:')
    print('python org_flash_block_make_figures_2.py phase_ramps_1 phase_ramps_2 phase_ramps_3...')
    print('\tor')
    print('python org_flash_block_make_figures_2.py phase_ramps_*')

if len(args)<2:
    usage()
    sys.exit()

for arg in args:
    if not os.path.exists(arg):
        continue
    test = glob.glob(os.path.join(arg,'*.npy'))
    if len(test)==0:
        continue
    folders.append(arg)

if len(folders)==0:
    sys.exit('Cannot find any .npy files in these folders: %s'%args)
    
os.makedirs(output_folder,exist_ok=True)


figure_list = []

def savefig(plot_type,file_tag):
    outdir = os.path.join(output_folder,plot_type)
    os.makedirs(outdir,exist_ok=True)
    outfn = os.path.join(outdir,'%s_%s.png'%(plot_type,file_tag))
    plt.savefig(outfn,dpi=print_dpi)
    figure_list.append((plot_type,file_tag,outfn))

opf.setup_plots(figure_mode)
color_cycle = opf.get_color_cycle()

def get_files(folder):
    return sorted(glob.glob(os.path.join(folder,'*.npy')))

file_lists = []
for folder in folders:
    file_lists.append(get_files(folder))

# verify that the supplied file lists are commensurate--same numbers of files, etc.
try:
    assert all(len(file_lists[0])==len(fl) for fl in file_lists)
    n_files = len(file_lists[0])
except AssertionError:
    sys.exit('Folders %s contain different numbers of files: %s.'%(folders,[len(fl) for fl in file_lists]))
    
for tup in zip(*file_lists):
    temp = [os.path.split(f)[1] for f in tup]
    for t1 in temp:
        for t2 in temp:
            try:
                assert t1==t2
            except AssertionError:
                sys.exit('Incomensurate files found in %s: %s.'%(folders,tup))


# use temporal sampling, number of files, and the file index of the stimulus
# to generate a time array
t_arr = (np.arange(n_files)-index_of_stimulus)*dt_s
stim_idx = np.argmin(np.abs(t_arr))

def nanmean(arr,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr,axis=axis)

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1.0500

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0

# load a single file to get the sz and sx values
temp = np.load(file_lists[0][0])
temp_amp = np.real(temp)
temp_vel = phase_to_nm(np.imag(temp))
sz,sx = temp.shape
z_arr = np.arange(sz)*dz_um
bscan_xlim = (bscan_crop_left,sx-bscan_crop_right)

def make_kernel(kernel_size,sigma,mode='gaussian'):
    XX,YY = np.meshgrid(np.arange(kernel_size),np.arange(kernel_size))
    XX = XX-sigma
    YY = YY-sigma
    rad = np.sqrt(XX**2+YY**2)
    kernel = np.zeros(rad.shape)
    if mode=='rect':
        kernel[np.where(rad<sigma)] = 1
        kernel = kernel/np.sum(kernel)
    elif mode=='gaussian':
        kernel = np.exp(-(rad**2)/(2*sigma**2))
        kernel = kernel/np.sum(kernel)
    return kernel

def scalebar_x(um_per_px,length_um=100,yoffset_px=10,xoffset_px=10+bscan_crop_left,ax=None,linewidth=5,color=color_cycle[-1],alpha=0.9):
    if ax is None:
        ax = plt.gca()
    x0 = xoffset_px
    x1 = x0+length_um/um_per_px
    y = yoffset_px
    ax.plot([x0,x1],[y,y],linewidth=linewidth,color=color,alpha=alpha)

def scalebar_y(um_per_px,length_um=100,yoffset_px=10,xoffset_px=10+bscan_crop_left,ax=None,linewidth=5,color=color_cycle[-1],alpha=0.9):
    if ax is None:
        ax = plt.gca()
    y0 = yoffset_px
    y1 = y0+length_um/um_per_px
    x = xoffset_px
    ax.plot([x,x],[y0,y1],linewidth=linewidth,color=color,alpha=alpha)

def scalebars(x_um_per_px,y_um_per_px,ax=None):
    scalebar_x(x_um_per_px,ax=ax)
    scalebar_y(y_um_per_px,ax=ax)

def show_bscan(b,screen_dpi=100,x_stretch=1.0,y_stretch=1.0,ax=None):
    sy,sx = b.shape
    fy = float(sy)/screen_dpi*y_stretch
    fx = float(sx)/screen_dpi*x_stretch*1.111

    scaling_factor = bscan_figure_width/fx
    fx = fx*scaling_factor
    fy = fy*scaling_factor
    
    fig = plt.figure(figsize=(fx,fy),dpi=screen_dpi)
    ax = fig.add_axes([0.02,0.02,.85,0.96])
    cax = fig.add_axes([0.89,0.1,0.02,0.8])
    db = 20*np.log10(b)
    clim = np.percentile(db,(5,99.9))
    imh = ax.imshow(db,clim=clim,cmap='gray',aspect='auto')
    ax.set_xlim(bscan_xlim)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(imh,cax=cax)
    return fig,ax

def show_bscan_overlay(amp,vel,screen_dpi=100,x_stretch=1.0,y_stretch=1.0,ax=None,alpha=0.5,extent=None):
    sy,sx = amp.shape
    fy = float(sy)/screen_dpi*y_stretch
    fx = float(sx)/screen_dpi*x_stretch*1.111
    
    scaling_factor = bscan_figure_width/fx
    fx = fx*scaling_factor
    fy = fy*scaling_factor
    
    fig = plt.figure(figsize=(fx,fy),dpi=screen_dpi)
    ax = fig.add_axes([0.02,0.02,.85,0.96])
    cax = fig.add_axes([0.89,0.1,0.02,0.8])
    db = 20*np.log10(amp)
    clim = np.percentile(db,(5,99.9))
    imh = ax.imshow(db,clim=clim,cmap='gray',aspect='auto',extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    vel_nan = np.zeros(vel.shape)
    vel_nan[:] = vel[:]
    vel_nan[np.where(vel==0)] = np.nan
    velh = ax.imshow(vel_nan,clim=bscan_vel_clim,cmap='jet',alpha=alpha,aspect='auto')
    ax.set_xlim(bscan_xlim)
    fig.colorbar(velh,cax=cax)
    return fig,ax


def linereg(a,b,mask=None):
    if mask is None:
        mask = np.ones(a.shape)

    aa = (a-a.mean())/a.std()
    bb = (b-b.mean())/b.std()
    xc = np.real(np.fft.ifft(np.fft.fft(aa*mask)*np.conj(np.fft.fft(bb*mask))))
    p = np.argmax(xc)
    if p>len(a)//2:
        p = p - len(a)
    return p

def get_contour0(b):
    hw = (flattening_averaging_width-1)//2
    sy,sx = b.shape
    out = []
    ref = b[:,sx//2-hw:sx//2+hw].mean(axis=1)
    for x in range(sx):
        x1 = x-hw
        x2 = x+hw+1
        while x1<0:
            x1+=1
        while x2>=sx:
            x2-=1
        p = linereg(ref,b[:,x1:x2].mean(1))
        out.append(p)
    out = np.array(out)
    out = sps.medfilt(out,3)
    x = np.arange(len(out))
    out = np.polyval(np.polyfit(x,out,1),x)
    return np.round(out).astype(np.int)

def get_contour(b,maxtilt=10):
    tmaxes = np.arange(-maxtilt,maxtilt+1)
    contrasts = np.zeros(tmaxes.shape)
    amaxes = np.zeros(tmaxes.shape)

    contours = [[]]*len(tmaxes)
    
    for idx,tmax in enumerate(tmaxes):
        tilt = np.round(np.linspace(0,tmax,b.shape[1])).astype(int)
        temp = np.zeros(b.shape)
        for x in range(b.shape[1]):
            temp[:,x] = np.roll(b[:,x],tilt[x])
        temp[:,:stimulated_region_start] = np.nan
        temp[:,stimulated_region_end:] = np.nan
        p = nanmean(temp,axis=1)
        contrast = (p.max()-p.min())/(p.max()+p.min())
        contrasts[idx] = contrast
        amaxes[idx] = p.max()
        contours[idx] = tilt

    return contours[np.argmax(contrasts)]
    

def roll_block(block,contour):
    sy,sz,sx = block.shape
    assert(len(contour)==sx)
    for x in range(sx):
        block[:,:,x] = np.roll(block[:,:,x],contour[x])
    return block

def make_blocks(folder,diagnostics=False):
    tag = make_tag(folder)

    file_list = get_files(folder)
    
    amp_block = []
    vel_block = []

    if vel_lateral_smoothing_sigma:
        vel_lateral_smoothing_kernel = make_kernel(vel_lateral_smoothing_sigma,vel_lateral_smoothing_sigma)

    
    # incoming data are stored in a weird way, due to upstream code:
    # the real component is amplitude and the imaginary component is
    # phase slope, in radians
    for f in file_list:
        temp = np.load(f)
        amp = np.real(temp)
        phase_slope = -np.imag(temp) # the negative sign is here because of the location of the zero-delay line
        vel = phase_to_nm(phase_slope)
        if vel_lateral_smoothing_sigma:
            vel = sps.convolve2d(vel,vel_lateral_smoothing_kernel,mode='same')/np.sum(vel_lateral_smoothing_kernel)
        amp_block.append(amp)
        vel_block.append(vel)

    amp_block = np.array(amp_block)
    vel_block = np.array(vel_block)
    
    if flatten:
        pre_contour_amp = nanmean(amp_block[stim_idx-20:stim_idx+20,:,:],axis=0)
        contour = get_contour(pre_contour_amp)
        
        amp_block = roll_block(amp_block,contour)
        post_contour_amp = nanmean(amp_block[stim_idx-20:stim_idx+20,:,:],axis=0)
        vel_block = roll_block(vel_block,contour)
        if diagnostics:
            plt.figure()
            show_bscan(pre_contour_amp)
            plt.title('pre flattening')
            show_bscan(post_contour_amp)
            plt.title('post_flattening')
            

    return amp_block,vel_block

def make_mask(im,fractional_threshold=0.1):
    out = np.ones(im.shape)*np.nan
    out[im>fractional_threshold*np.nanmax(im)] = 1.0
    return out

def get_peak_dict(prof):
    left = prof[2:]
    center = prof[1:-1]
    right = prof[:-2]
    peaks = np.where((center>left)*(center>right)*(center>profile_peak_threshold))[0]
    peaks = peaks + 1
    d = {}
    for idx,peak in enumerate(peaks):
        try:
            d[peak_labels[idx]] = peak
        except:
            d['peak_%d'%idx] = peak
    return d

def make_profile_plot(prof,peak_dict):
    plt.figure(figsize=profile_figure_size,dpi=screen_dpi)
    plt.plot(prof)
    for kidx,key in enumerate(peak_dict.keys()):
        c = color_cycle[kidx%len(color_cycle)]
        idx = peak_dict[key]
        plt.axvline(idx,color=c)
        plt.text(idx,prof[idx],key,ha='right',va='bottom',rotation=90,color=c,fontsize=6)
    plt.yticks([])
    plt.ylabel('OCT amplitude')
    opf.despine()

def text(ax,x,y,s,ha='right',va='center',color='w'):
    h = 20
    w = 6.5*len(s)
    ax.text(x-2,y,s,ha=ha,va=va,color='k',fontsize=6)
    
def add_labels(ax,pd,xlim=None):
    if xlim is None:
        xlim = plt.gca().get_xlim()

    for k in pd.keys():
        if k in peak_labels:
            text(ax,xlim[0],pd[k],k)
            
def show_mscan_overlay(amp_m,vel_m,screen_dpi=100,x_stretch=3.0,y_stretch=1.0,ax=None,alpha=0.5,mscan_xlim=tlim_ms,peak_dict={}):

    vel_m[:,:15] = np.nan

    #for y in range(sy):
    mask = make_mask(amp_m)

    sy,sx = amp_m.shape
    fy = float(sy)/screen_dpi*y_stretch*1.4
    fx = float(sx)/screen_dpi*x_stretch*1.111

    scaling_factor = bscan_figure_width/fx
    fx = fx*scaling_factor
    fy = fy*scaling_factor
    
    fig = plt.figure(figsize=(fx,fy),dpi=screen_dpi)
    ax = fig.add_axes([0.02,0.15,.85,0.75])
    cax = fig.add_axes([0.89,0.1,0.02,0.8])
    db = 20*np.log10(amp_m)
    clim = np.percentile(db,(5,99.9))
    imh = ax.imshow(db,clim=clim,cmap='gray',aspect='auto',extent=[1000*t_arr[0],1000*t_arr[-1],z_arr[0],z_arr[-1]])
    vh = ax.imshow(vel_m,clim=mscan_vel_clim,cmap='jet',aspect='auto',alpha=0.5,extent=[1000*t_arr[0],1000*t_arr[-1],z_arr[0],z_arr[-1]])

    ax.axvline(0.0,color=stim_color,linestyle=stim_linestyle)
    ax.set_xlim(mscan_xlim)
    ax.set_xlabel('time (ms)')
    ax.set_title(r'dz/dt ($\mu$m/s)')
    ax.set_yticks([])
    for peak_label in peak_labels:
        try:
            zpx = peak_dict[peak_label]
            zum = zpx*dz_um
            zum = z_arr[-1]-zum
            ax.text(mscan_xlim[0],zum,peak_label,ha='left',va='center')
        except:
            pass
            
    fig.colorbar(vh,cax=cax)

    
    
#### block functions

def get_amp(amp_block):
    bscan = nanmean(amp_block[stim_idx-10:stim_idx+10,:,:],axis=0)
    return bscan

def get_vel(vel_block,t0=0.0,t1=0.01):
    idx1 = np.argmin(np.abs(t_arr-t0))
    idx2 = np.argmin(np.abs(t_arr-t1))
    bscan = nanmean(vel_block[idx1:idx2,:,:],axis=0)
    return bscan

def get_roi(vel_block,x1,x2,z1,z2):
    sub = vel_block[:,z1:z2,x1:x2]
    sub = nanmean(nanmean(sub,axis=2),axis=1)
    return sub

def make_tag(path):
    toks = []
    while True:
        tup = os.path.split(path)
        toks = [tup[1]]+toks
        path = tup[0]
        if len(path)==0:
            break
    return '_'.join(toks)

def add_origin():
    alpha = 0.5
    linestyle=':'
    color=[0.5,0.5,0.5]
    plt.axvline(0.0,linestyle=linestyle,color=color,alpha=alpha)
    plt.axhline(0.0,linestyle=linestyle,color=color,alpha=alpha)

layer_dict = {}

for folder_idx,folder in enumerate(folders):
    logging.info('Working on folder %d of %d: %s.'%(folder_idx+1,len(folders),folder))
    tag = make_tag(folder)
    file_list = get_files(folder)
    ablock,vblock = make_blocks(folder)
    sy,sz,sx = ablock.shape
    bscan = get_amp(ablock)

    mask = make_mask(bscan)
    vblock = vblock*mask
    
    prestim = get_vel(vblock,-0.03,0.0)
    poststim = get_vel(vblock,0.00,0.020)
    
    prof = nanmean(bscan[:,stimulated_region_start:stimulated_region_end],axis=1)
    peak_dict = get_peak_dict(prof)

    plt.figure(figsize=plots_figure_size,dpi=screen_dpi)
    for peak_label in peak_labels:
        z = peak_dict[peak_label]
        series = get_roi(vblock,stimulated_region_start,stimulated_region_end,z-layer_thickness//2,z+layer_thickness//2+1)
        series = series - nanmean(series[:stim_idx])
        try:
            layer_dict[peak_label].append(series)
        except KeyError:
            layer_dict[peak_label] = [series]
            
        plt.plot(1000*t_arr,series,label=peak_label)
    add_origin()
    
    plt.xlabel('time (ms)')
    plt.ylabel('velocity ($\mu m/s$)')
    plt.ylim(abs_plot_clim)
    plt.xlim(tlim_ms)
    opf.despine()
    plt.legend(frameon=False)
    savefig('layer_velocities',tag)

    plt.figure(figsize=plots_figure_size,dpi=screen_dpi)
    for a,b in layer_differences:
        dseries = layer_dict[a][-1]-layer_dict[b][-1]
        plt.plot(1000*t_arr,dseries,label='%s - %s'%(a,b))
    add_origin()
    plt.xlabel('time (ms)')
    plt.ylabel('velocity ($\mu m/s$)')
    plt.ylim(rel_plot_clim)
    plt.xlim(tlim_ms)
    opf.despine()
    plt.legend(frameon=False)
    savefig('layer_velocity_differences',tag)
        
    make_profile_plot(prof,peak_dict)
    plt.axhline(profile_peak_threshold)
    plt.text(0,profile_peak_threshold,'peak threshold',ha='left',va='bottom')
    opf.despine()
    savefig('peak_profiles',tag)

    fig,ax = show_bscan(bscan)
    savefig('bscan_amp',tag)
    scalebars(3.0,2.5,ax=ax)
    fig,ax = show_bscan_overlay(bscan,prestim)
    scalebars(3.0,2.5,ax=ax)
    savefig('bscan_pre_stim',tag)
    fig,ax = show_bscan_overlay(bscan,poststim)
    scalebars(3.0,2.5,ax=ax)
    savefig('bscan_post_stim',tag)


    amp_m = nanmean(ablock[:,:,stimulated_region_start:stimulated_region_end],axis=2).T
    vel_m = nanmean(vblock[:,:,stimulated_region_start:stimulated_region_end],axis=2).T
    show_mscan_overlay(amp_m,vel_m,peak_dict=peak_dict)
    savefig('mscan',tag)
    plt.close('all')


logging.info('Working on average plots.')
plt.figure(figsize=plots_figure_size,dpi=screen_dpi)
for peak_label in peak_labels:
    arr = np.array(layer_dict[peak_label])
    avg = nanmean(arr,axis=0)
    plt.plot(1000*t_arr,avg,label=peak_label)
add_origin()
plt.xlabel('time (ms)')
plt.ylabel('velocity ($\mu m/s$)')
plt.ylim(abs_plot_clim)
plt.xlim(tlim_ms)
opf.despine()
plt.legend(frameon=False)
savefig('layer_velocities','average')

plt.figure(figsize=plots_figure_size,dpi=screen_dpi)
for a,b in layer_differences:
    a_arr = np.array(layer_dict[a])
    b_arr = np.array(layer_dict[b])
    a_avg = nanmean(a_arr,axis=0)
    b_avg = nanmean(b_arr,axis=0)
    d = a_avg-b_avg
    plt.plot(1000*t_arr,d,label='%s - %s'%(a,b))
add_origin()
plt.xlabel('time (ms)')
plt.ylabel('velocity ($\mu m/s$)')
plt.ylim(rel_plot_clim)
plt.xlim(tlim_ms)
opf.despine()
plt.legend(frameon=False)
savefig('layer_velocity_differences','average')


logging.info('Writing report.')
figure_list.sort(key=lambda tup: tup[0])


d = {}
d['bscan_amp'] = 'Amplitude B-scans'
d['bscan_pre'] = 'Velocity (pre stimulus)'
d['bscan_post'] = 'Velocity (post stimulus)'
d['mscan'] = 'M-scans'
d['layer_velocities'] = 'Layer velocity plots'
d['layer_velocity_differences'] = 'Relative layer velocity plots'

if make_pdf:
    try:
        with open('%s/figures.md'%output_folder,'w') as fid:
            for label in ['bscan_amp','bscan_pre','bscan_post','mscan','layer_velocities','layer_velocity_differences']:
                sublist = [f for f in figure_list if f[0].find(label)>-1]
                sublist.sort(key=lambda tup: tup[0])
                fid.write('### %s\n\n'%d[label])
                for plot_type,file_tag,outfn in sublist:
                    outfn_list = outfn.split('/')
                    outfn = '/'.join(outfn_list[1:])
                    fid.write('![%s: %s](%s)\n\n'%(plot_type,file_tag,outfn))
                fid.write('---\n\n')
                
        os.system('cd %s && pandoc -V geometry:margin=0.5in -o figures.pdf figures.md'%output_folder)
        os.system('cd ..')
    except Exception as e:
        print(e)

try:
    fign = 1
    secn = 1
    with open('%s/figures.html'%output_folder,'w') as fid:
        fid.write('<head><title>Conventional flash ORG figures</title></head>\n')
        fid.write('<body>\n')
        for label in ['bscan_amp','bscan_pre','bscan_post','mscan','layer_velocities','layer_velocity_differences']:
            sublist = [f for f in figure_list if f[0].find(label)>-1]
            sublist.sort(key=lambda tup: tup[0])
            fid.write('<h3>%d. %s</h3>\n\n'%(secn,d[label]))
            secn+=1
            for plot_type,file_tag,outfn in sublist:
                outfn_list = []
                while len(outfn)>0:
                    outfn_list.append(os.path.split(outfn)[1])
                    outfn = os.path.split(outfn)[0]
                outfn_list = outfn_list[::-1]
                outfn = ''
                for item in outfn_list[1:]:
                    outfn = os.path.join(outfn,item)                
                fid.write('<p>')
                
                fid.write('<a href=\"%s\">\n'%outfn)
                fid.write('<img src=\"%s\" alt=\"%s: %s\" width=\"33%%\">\n'%(outfn,plot_type,file_tag))
                fid.write('</a>\n')
                fid.write('</p>\n')
                fid.write('<p>Fig. %d. %s / %s</p>\n'%(fign,plot_type,file_tag))
                fign+=1
            fid.write('<hr>\n\n')
        fid.write('</body>')
except Exception as e:
    print(e)

try:
    if auto_open_report:
        report_fn = os.path.join(output_folder,'figures.html')
        wb.open(report_fn)
        #os.system('firefox %s/figures.html'%output_folder)
except:
    pass
