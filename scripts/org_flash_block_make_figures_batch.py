import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob,os,sys,shutil
import scipy.signal as sps
import scipy.io as sio
import octoblob.plotting_functions as opf
import warnings
import octoblob as blob
import webbrowser as wb
from difflib import SequenceMatcher
from org_flash_block_summary import summary_datafile
import pandas as pd
import pathlib

################################### Parameters ##########################################


# parameters for now, but turn these into command line arguments
figure_mode = 'paper' # 'paper' or 'presentation'

# specify the file containing the onset of the stimulus; it will be identified by finding
# the first filename containing this substring in the alphanumerically sorted files:
stimulus_file_filter = 'phase_ramp_frames_00100'

dx_um = 3.0
dy_um = 3.0
dz_um = 2.5
dt_s = 0.0025

# error threshold--phase fitting error ranges from about 0 rad to 0.5 rad; a reasonable
# threshold is 0.05; to include all data use err_threshold = -np.inf
err_threshold = -np.nan

# the duration over which we assume the retina is stationary (in seconds)
stationary_duration = 0.01

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
# None in either place defaults to the start or end of the existing ramp files
tlim_ms = (-100,100)
# tlim_ms = (None,None)

# crop b-scans, just for visualization
bscan_crop_left = 10
bscan_crop_right = 20

vel_lateral_smoothing_sigma = 3 # use 0 for no smoothing

# when refining the localization of peaks, allow individual
# peak positions to vary slightly from the peaks in the mean
# profile, to accommodate slight variations in fixation/eccentricity:
axial_peak_shift_tolerance = 2

profile_peak_threshold = 5000
peak_labels = ['ISOS','COST']
layer_differences = [('COST','ISOS')]#,('RPE','ISOS')]

# flattening of the data volumes in the fast and temporal dimensions:
flatten_bscan = True # flatten B-scans using 1D A-scan registration
flattening_averaging_width = 5 # number of A-scans to average for flattening
flatten_mscan = True

# B-scans are flattened incorrectly if the NFL or ILM are brighter than the outer
# retinal bands, since the flatness of those layers becomes the key figure of merit
# if set to True, all peaks inside a boundary are ignored during flattening; the
# boundary is set by computing the axial center of mass and offsetting it:
ignore_inner_peaks = True
ignore_inner_peaks_com_offset = -10

# figure apperance
figure_size = (4,3) # (width_inches, height_inches)
font = 'Arial'
font_size = 10
screen_dpi = 100
print_dpi = 300
plot_linewidth = 2

# see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
#style = 'bmh'
style = 'ggplot'
#style = 'seaborn-deep'
#style = 'fivethirtyeight'

# options for plotting raw data and/or noise in background of average:
plot_error_region = True
plot_single_measurements = True
single_color = 'r'
single_alpha = 0.1
single_linewidth = 1
noise_alpha = 0.1
noise_color = 'k'

# spines (lines at edges of plot)
spine_color = 'k'
spine_linewidth = 2

# legend appearance
show_legend_frame = True
legend_linewidth = 2
legend_edgecolor = 'k'
legend_facecolor = 'w'
legend_alpha = 1.0

# Indiviual layers are automatically plotted in colors in order
# of the style's color_cycle; for the difference plots, choose
# an index (greater or equal to the number of layers), so that the difference
# plot appears in a different color
# The actual color that results depends on style, defined below. If
# the style is ggplot, for example, the first two layers (ISOS and COST) are
# plotted in red and blue, and here 5 gives a nice, contrasting green/yellow color
layer_difference_color_index = 5

velocity_label = 'μm/s'
#velocity_label = '$\mu m/s$'


stim_color = 'g'
stim_linestyle = '-'


output_folder = 'org_block_figures'
auto_open_report = True
make_pdf = False # requires pandoc
testing_length_differences = False

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


data_dictionary = {}
def add_to_data_dictionary(s1,s2,dat):
    key = '%s_%s'%(s1,s2)
    data_dictionary[key] = dat



styles_with_origin = ['ggplot','fivethirtyeight','bmh']
opf.setup_plots(style=style,font_size=font_size,font=font)
color_cycle = opf.get_color_cycle()


if summary_datafile=='':
    sys.exit('Please enter a location for a summary data file in org_flash_block_summary.py.')


summary_columns = ['filename','date','time','stationary_duration','layers','vmin_0_20','vmax_20_40','vmean_20_40','amin_0_50','amax_0_50','std_0_50','mad_0_50']
for ld in layer_differences:
    summary_columns.append('%s_%s_px'%(ld[0],ld[1]))


if not os.path.exists(summary_datafile):
    summary_df = pd.DataFrame(columns=summary_columns)
else:
    summary_df = pd.read_csv(summary_datafile,index_col=0)

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
    ax = plt.gca()
    ax.tick_params(direction='in')
    ax.tick_params(left=True)
    ax.tick_params(right=True)
    ax.tick_params(top=True)
    ax.tick_params(bottom=True)

    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(spine_linewidth)
    
    plt.tight_layout()
    outdir = os.path.join(output_folder,plot_type)
    os.makedirs(outdir,exist_ok=True)
    outfn = os.path.join(outdir,'%s_%s.png'%(plot_type,file_tag))
    plt.savefig(outfn,dpi=print_dpi)
    svgoutfn = os.path.join(outdir,'%s_%s.svg'%(plot_type,file_tag))
    plt.savefig(svgoutfn,dpi=print_dpi)
    
    figure_list.append((plot_type,file_tag,outfn))

#opf.setup_plots(figure_mode)
color_cycle = opf.get_color_cycle()

def get_files(folder):
    return sorted(glob.glob(os.path.join(folder,'*.npy')))

file_lists = []
for folder in folders:
    file_lists.append(get_files(folder))


metatag = ''

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

temp = file_lists[0]
index_of_stimulus=None
for idx,f in enumerate(temp):
    if f.find(stimulus_file_filter)>-1:
        index_of_stimulus=idx
        
if index_of_stimulus is None:
    sys.exit('stimulus_file_filter %s did not identify a stimulus frame'%stimulus_file_filter)


t_pre_stim = (index_of_stimulus-1)*dt_s-stationary_duration/2.0

# use temporal sampling, number of files, and the file index of the stimulus
# to generate a time array
t_arr = np.arange(n_files)*dt_s-t_pre_stim#index_of_stimulus)*dt_s

stim_idx = np.argmin(np.abs(t_arr))

new_start = tlim_ms[0]
new_end = tlim_ms[1]
if new_start is None:
    new_start = t_arr[0]*1000.0
if new_end is None:
    new_end = t_arr[-1]*1000.0

tlim_ms = (new_start,new_end)


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

def show_bscan(b,screen_dpi=100,x_stretch=1.0,y_stretch=1.0,ax=None,layer_dict={}):
    # sy,sx = b.shape
    # fy = float(sy)/screen_dpi*y_stretch
    # fx = float(sx)/screen_dpi*x_stretch*1.111

    # scaling_factor = bscan_figure_width/fx
    # fx = fx*scaling_factor
    # fy = fy*scaling_factor
    
    fig = plt.figure(figsize=figure_size,dpi=screen_dpi)
    ax = fig.add_axes([0.02,0.02,.85,0.96])
    cax = fig.add_axes([0.89,0.1,0.02,0.8])
    db = 20*np.log10(b)
    clim = np.percentile(db,(5,99.9))
    imh = ax.imshow(db,clim=clim,cmap='gray',aspect='auto')
    ax.set_xlim(bscan_xlim)
    ax.set_xticks([])
    ax.set_yticks([])

    for idx,label in enumerate(layer_dict.keys()):
        color = color_cycle[idx%len(color_cycle)]
        z = layer_dict[label]
        ax.axhline(z,color=color,alpha=0.5)
        ax.text(bscan_xlim[0],z,label,ha='left',va='bottom',color=color)
    
    fig.colorbar(imh,cax=cax)
    return fig,ax

def show_bscan_overlay(amp,vel,screen_dpi=100,x_stretch=1.0,y_stretch=1.0,ax=None,alpha=0.5,extent=None):
    # sy,sx = amp.shape
    # fy = float(sy)/screen_dpi*y_stretch
    # fx = float(sx)/screen_dpi*x_stretch*1.111
    
    # scaling_factor = bscan_figure_width/fx
    # fx = fx*scaling_factor
    # fy = fy*scaling_factor
    
    fig = plt.figure(figsize=figure_size,dpi=screen_dpi)
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

def get_contour(b,maxtilt=20,tilt_step=0.1,diagnostics=False):
    b_filtered = np.zeros(b.shape)
    
    if ignore_inner_peaks:
        p = np.nanmean(b,axis=1)
        p = p-np.nanmin(p)
        z = np.arange(len(p))
        com = int(np.floor(np.sum(p*z)/np.sum(p)))
        cutoff = com+ignore_inner_peaks_com_offset
        b_filtered[cutoff:,:] = b[cutoff:,:]
    else:
        b_filtered[:,:] = b[:,:]
    
        
    tmaxes = np.arange(-maxtilt,maxtilt+1,tilt_step)
    contrasts = np.ones(tmaxes.shape)*np.nan
    amaxes = np.ones(tmaxes.shape)*np.nan
    
    contours = [[]]*len(tmaxes)

    if diagnostics:
        flattened_profiles = np.zeros((b.shape[0],len(tmaxes)))
        plt.figure(figsize=(6,4))
        
    for idx,tmax in enumerate(tmaxes):
        tilt = np.round(np.linspace(0,tmax,b_filtered.shape[1])).astype(int)
        temp = np.zeros(b_filtered.shape)
        for x in range(b_filtered.shape[1]):
            temp[:,x] = np.roll(b_filtered[:,x],tilt[x])
        temp[:,:stimulated_region_start] = np.nan
        temp[:,stimulated_region_end:] = np.nan

        temp[np.where(temp==0)] = np.nan

        
        p = nanmean(temp,axis=1)

        contrast = (np.nanmax(p)-np.nanmin(p))/(np.nanmax(p)+np.nanmin(p))
        #contrast = (p.max()-p.min())/(p.max()+p.min())
        contrasts[idx] = contrast
        amaxes[idx] = np.nanmax(p)
        contours[idx] = tilt
        
        if diagnostics:
            flattened_profiles[:,idx] = p
            if idx%10==0:
                plt.clf()

                plt.subplot(2,2,1)
                plt.cla()
                plt.imshow(flattened_profiles,cmap='gray',aspect='auto')
                plt.colorbar()

                plt.subplot(2,2,2)
                plt.cla()
                plt.imshow(20*np.log10(temp),cmap='gray',clim=(40,90),aspect='auto')
                plt.colorbar()


                plt.subplot(2,2,3)
                plt.cla()
                plt.plot(tmaxes,contrasts)
                plt.xlabel('shear')
                plt.ylabel('contrast')

                plt.subplot(2,2,4)
                plt.cla()
                plt.plot(tmaxes,amaxes)
                plt.xlabel('shear')
                plt.ylabel('profile max')

                plt.pause(.01)

    if diagnostics:
        plt.show()
        
    return contours[np.argmax(contrasts)]
    

def roll_block(block,contour):
    sy,sz,sx = block.shape
    assert(len(contour)==sx)
    for x in range(sx):
        block[:,:,x] = np.roll(block[:,:,x],contour[x])
    return block

def make_blocks(folder,diagnostics=False):
    tag,stag = make_tag(folder)

    err_folder = folder.replace('phase_ramps','err')

    
    file_list = get_files(folder)

    err_file_list = get_files(err_folder)
    
    amp_block = []
    vel_block = []
    err_block = []
    
    if vel_lateral_smoothing_sigma:
        vel_lateral_smoothing_kernel = make_kernel(vel_lateral_smoothing_sigma,vel_lateral_smoothing_sigma)

    
    # incoming data are stored in a weird way, due to upstream code:
    # the real component is amplitude and the imaginary component is
    # phase slope, in radians
    for fidx,f in enumerate(file_list):
        
        temp = np.load(f)
        amp = np.real(temp)
        phase_slope = -np.imag(temp) # the negative sign is here because of the location of the zero-delay line
        vel = phase_to_nm(phase_slope)
        if vel_lateral_smoothing_sigma:
            vel = sps.convolve2d(vel,vel_lateral_smoothing_kernel,mode='same')/np.sum(vel_lateral_smoothing_kernel)
        amp_block.append(amp)
        vel_block.append(vel)
        try:
            err_block.append(np.load(err_file_list[fidx]))
        except:
            err_block.append(np.zeros(amp.shape))
        
    amp_block = np.array(amp_block)
    vel_block = np.array(vel_block)
    err_block = np.array(err_block)
    
    if flatten_bscan:
        pre_contour_amp = nanmean(amp_block[stim_idx-20:stim_idx+20,:,:],axis=0)
        contour = get_contour(pre_contour_amp)
        
        amp_block = roll_block(amp_block,contour)
        post_contour_amp = nanmean(amp_block[stim_idx-20:stim_idx+20,:,:],axis=0)
        
        vel_block = roll_block(vel_block,contour)
        err_block = roll_block(err_block,contour)
        if diagnostics:
            plt.figure()
            show_bscan(pre_contour_amp)
            plt.title('pre flattening')
            show_bscan(post_contour_amp)
            plt.title('post_flattening')

    if flatten_mscan:
        xproj = nanmean(amp_block,axis=2).T
        ref = xproj[:,stim_idx-5:stim_idx+5].mean(axis=1)

        oversample = 5.0
        tcontour = []
        for x in range(xproj.shape[1]):
            tar = xproj[:,x]
            xc = np.real(np.fft.ifft(np.fft.fft(ref)*np.conj(np.fft.fft(tar)),n=int(len(ref)*oversample)))
            p = np.argmax(xc)
            if p>len(xc)//2:
                p = p - len(xc)
            tcontour.append(int(round(p/oversample)))

        temp = []
        for x in range(len(tcontour)):
            pre = np.zeros((amp_block.shape[1],amp_block.shape[2]))
            post = np.zeros((amp_block.shape[1],amp_block.shape[2]))
            vel_block[x,:,:] = np.roll(vel_block[x,:,:],tcontour[x],axis=0)
            amp_block[x,:,:] = np.roll(amp_block[x,:,:],tcontour[x],axis=0)
            err_block[x,:,:] = np.roll(err_block[x,:,:],tcontour[x],axis=0)
            
            

        temp = np.array(temp)
        
        xproj = nanmean(amp_block,axis=2).T
    return amp_block,vel_block,err_block



def make_mask(im,fractional_threshold=0.1):
    out = np.ones(im.shape)*np.nan
    out[im>fractional_threshold*np.nanmax(im)] = 1.0
    return out

def format_legend(leg):
    
    leg.get_frame().set_edgecolor(legend_edgecolor)
    leg.get_frame().set_facecolor(legend_facecolor)
    leg.get_frame().set_linewidth(legend_linewidth)
    leg.get_frame().set_alpha(legend_alpha)

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
    fig = plt.figure(figsize=figure_size,dpi=screen_dpi)

    ax = fig.add_axes([.2,.2,.78,.78])
    ax.plot(prof)
    for kidx,key in enumerate(peak_dict.keys()):
        c = color_cycle[kidx%len(color_cycle)]
        idx = peak_dict[key]
        ax.axvline(idx,color=c)
        ax.text(idx,prof[idx],key,ha='right',va='bottom',rotation=90,color=c,fontsize=6)
    #plt.yticks([])
    ax.set_ylabel('OCT amplitude')
    ax.set_xlabel('depth (px)')
    #opf.despine()

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
            
def show_mscan_overlay(amp_m,vel_m,screen_dpi=100,x_stretch=3.0,y_stretch=1.0,ax=None,alpha=0.5,mscan_xlim=tlim_ms,peak_dict={},do_overlay=True,blank_left_edge=False):

    if blank_left_edge:
        vel_m[:,:15] = np.nan

    #for y in range(sy):
    mask = make_mask(amp_m)

    fig = plt.figure(figsize=figure_size,dpi=screen_dpi)
    ax = fig.add_axes([0.02,0.15,.85,0.75])
    if do_overlay:
        cax = fig.add_axes([0.89,0.1,0.02,0.8])
    db = 20*np.log10(amp_m)
    clim = np.percentile(db,(5,99.9))
    
    imh = ax.imshow(db,clim=clim,cmap='gray',aspect='auto',extent=[1000*t_arr[0],1000*t_arr[-1],z_arr[0],z_arr[-1]])
    
    if do_overlay:
        vh = ax.imshow(vel_m,clim=mscan_vel_clim,cmap='jet',aspect='auto',alpha=0.5,extent=[1000*t_arr[0],1000*t_arr[-1],z_arr[0],z_arr[-1]])

    ax.axvline(0.0,color=stim_color,linestyle=stim_linestyle)
    ax.set_xlim(mscan_xlim)
    ax.set_xlabel('time (ms)')
    ax.set_title(r'dz/dt (%s)'%velocity_label)
    ax.set_yticks([])
    ax.grid(False)

    if blank_left_edge:
        for peak_label in peak_labels:
            try:
                zpx = peak_dict[peak_label]
                zum = zpx*dz_um
                zum = z_arr[-1]-zum
                ax.text(mscan_xlim[0],zum,peak_label,ha='left',va='center')
            except:
                pass

    if do_overlay:
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


def all_ints(items):
    for item in items[:1]:
        try:
            junk = int(item)
            assert junk<100
        except:
            return False
    return True

def is_list_of_ints(L):
    for item in L:
        try:
            junk = int(item)
        except:
            return False
    return True
    

def make_tag(path):
    toks = []
    short_toks = []
    
    while True:
        tup = os.path.split(path)
        temp = tup[1].split('_')
        if all_ints(temp):
            cand = tup[1][:8]
            if not cand in short_toks:
                short_toks.append(cand)
        
        toks = [tup[1]]+toks
        path = tup[0]
        if len(path)==0:
            break
    return '_'.join(toks),'_'.join(short_toks)


def find_date(path,max_iterations=50):
    head = path
    count = 0
    while len(head)>0:
        head,tail = os.path.split(head)
        toks = tail.split('.')
        if is_list_of_ints(toks) and len(toks)==3:
            if int(toks[0])>2000 and int(toks[1])<=12 and int(toks[2])<=31:
                return tail
        count = count + 1
        if count>=max_iterations:
            sys.exit('Could not find a date of the form 2XXX.XX.XX in %s.'%path)
    
    return None

def add_origin():
    alpha = 0.5
    linestyle=':'
    color=[0.5,0.5,0.5]
    plt.axvline(0.0,linestyle=linestyle,color=color,alpha=alpha)
    plt.axhline(0.0,linestyle=linestyle,color=color,alpha=alpha)

layer_dict = {}
tags = []
stags = []

profs = []

if testing_length_differences:
    tempdict = {'12_45_49_bscans/cropped/phase_ramps_010ms_npy':125,
                '12_53_09_bscans/cropped/phase_ramps_010ms_npy':129,
                '12_58_29_bscans/cropped/phase_ramps_010ms_npy':123}


for folder_idx,folder in enumerate(folders):
    logging.info('Computing axial profile plots for set %d of %d: %s.'%(folder_idx+1,len(folders),folder))
    tag,stag = make_tag(folder)
    tags.append(tag)
    stags.append(stag)
    file_list = get_files(folder)
    ablock,vblock,eblock = make_blocks(folder,diagnostics=False)

    if testing_length_differences:
        ablock = ablock[:,:tempdict[folder],:]
    sy,sz,sx = ablock.shape
    bscan = get_amp(ablock)
    prof = nanmean(bscan[:,stimulated_region_start:stimulated_region_end],axis=1)
    profs.append(prof)

metatag = '_'.join(stags)


# reconcile prof lengths
max_length = np.max([len(prof) for prof in profs])

newprofs = []
for prof in profs:
    newprof = np.zeros(max_length)
    newprof[:len(prof)] = prof[:]
    newprofs.append(newprof)

profs = newprofs
    
prof_axial_shifts = []
fref = np.fft.fft(profs[0])
mprof = np.zeros(profs[0].shape)

for folder_idx,(tar,folder) in enumerate(zip(profs,folders)):
    logging.info('Registering axial profile plots for set %d of %d: %s.'%(folder_idx+1,len(folders),folder))

    ftar = np.fft.fft(tar)
    xc = np.real(np.fft.ifft(fref*np.conj(ftar)/np.abs(fref*np.conj(ftar))))
    pidx = np.argmax(xc)
    if pidx>len(xc)//2:
        pidx = pidx - len(xc)
    prof_axial_shifts.append(pidx)
    shifted = np.roll(tar,pidx)
    mprof = mprof + shifted

mprof = mprof/len(profs)
mpeak_dict = get_peak_dict(mprof)


peak_metadict = {}
for folder_idx,(prof,shift,folder) in enumerate(zip(profs,prof_axial_shifts,folders)):
    logging.info('Identifying peaks for set %d of %d: %s.'%(folder_idx+1,len(folders),folder))
    peak_dict = {}
    for k in mpeak_dict.keys():
        estimate = mpeak_dict[k]-shift
        peak_height = prof[estimate]
        peak_position = estimate
        for dz in range(-axial_peak_shift_tolerance,axial_peak_shift_tolerance+1):
            if prof[estimate+dz]>peak_height:
                peak_position = estimate+dz
                peak_height = prof[peak_position]
            if prof[peak_position]>prof[peak_position-1] and prof[peak_position]>prof[peak_position+1]:
                break
        peak_dict[k] = peak_position
    peak_metadict[folder] = peak_dict


logging.info('Dictionaries of peaks for all folders:')
for k in peak_metadict.keys():
    pd = peak_metadict[k]
    logging.info(k)
    for sk in pd.keys():
        logging.info('\t%s:\t%d'%(sk,pd[sk]))

        
for folder_idx,folder in enumerate(folders):
    logging.info('Making plots for folder %d of %d: %s.'%(folder_idx+1,len(folders),folder))
    tag,stag = make_tag(folder)
    tags.append(tag)

    summary_dict = {}
    
    cwd = str(pathlib.Path(__file__).parent.resolve())
    
    date = find_date(folder)
    if date is None:
        date = find_date(cwd)

    summary_dict['filename'] = os.path.join(cwd,folder)
    summary_dict['date'] = date
    summary_dict['time'] = stag
    summary_dict['stationary_duration'] = stationary_duration
    
    file_list = get_files(folder)
    ablock,vblock,eblock = make_blocks(folder)

    sy,sz,sx = ablock.shape
    bscan = get_amp(ablock)

    mask = make_mask(bscan)
    vblock = vblock*mask
    
    prestim = get_vel(vblock,-0.03,0.0)
    poststim = get_vel(vblock,0.00,0.020)
    
    prof = nanmean(bscan[:,stimulated_region_start:stimulated_region_end],axis=1)

    # old way: separate peaks for each profile:
    #peak_dict = get_peak_dict(prof)
    # new way: use peak metadictionary
    peak_dict = peak_metadict[folder]

    
    
    thickness_dict = {}
    for a,b in layer_differences:
        key = '%s_%s_px'%(a,b)
        dpx = peak_dict[a]-peak_dict[b]
        thickness_dict[key] = dpx
    
    plt.figure(figsize=figure_size,dpi=screen_dpi)
    for peak_label in peak_labels:
        z = peak_dict[peak_label]
        series = get_roi(vblock,stimulated_region_start,stimulated_region_end,z-layer_thickness//2,z+layer_thickness//2+1)

        err_series = get_roi(eblock,stimulated_region_start,stimulated_region_end,z-layer_thickness//2,z+layer_thickness//2+1)

        series = series - nanmean(series[stim_idx-10:stim_idx])

        series[np.where(err_series<err_threshold)] = np.nan
        
        try:
            layer_dict[peak_label].append(series)
        except KeyError:
            layer_dict[peak_label] = [series]
            
        plt.plot(1000*t_arr,series,label=peak_label,linewidth=plot_linewidth)
        add_to_data_dictionary(tag,'layer_velocity_%s'%peak_label,series)

    add_to_data_dictionary(tag,'t',1000*t_arr)
    
    if not style in styles_with_origin:
        add_origin()
    
    plt.xlabel('time (ms)')
    plt.ylabel('velocity (%s)'%velocity_label)
    plt.ylim(abs_plot_clim)
    plt.xlim(tlim_ms)
    #opf.despine()
    leg = plt.legend(frameon=show_legend_frame)
    format_legend(leg)
    savefig('layer_velocities',tag)

    plt.figure(figsize=figure_size,dpi=screen_dpi)
    for a,b in layer_differences:
        dseries = layer_dict[a][-1]-layer_dict[b][-1]
        if a=='COST' and b=='ISOS':
            layer_label = '$v_{OS}$'
        else:
            layer_label = '%s - %s'%(a,b)
        plt.plot(1000*t_arr,dseries,label=layer_label,color=color_cycle[layer_difference_color_index%len(color_cycle)],linewidth=plot_linewidth)
        add_to_data_dictionary(tag,'layer_velocity_difference_%s_%s'%(a,b),dseries)


    vmin_0_20 = np.nanmin(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.02))])
    vmax_20_40 = np.nanmax(dseries[np.where(np.logical_and(t_arr>=0.02,t_arr<=0.04))])
    vmean_20_40 = np.nanmean(dseries[np.where(np.logical_and(t_arr>=0.02,t_arr<=0.04))])
    amin_0_50 = np.nanmin(np.diff(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.05))]))
    amax_0_50 = np.nanmax(np.diff(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.05))]))
    std_0_50 = np.nanstd(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.05))])
    mad_0_50 = np.nanmean(np.abs(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.05))]))
    #mad_0_50 = np.nanmean(np.abs(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.05))]-np.nanmean(dseries[np.where(np.logical_and(t_arr>=0,t_arr<=0.05))])))
    
    summary_dict['vmin_0_20'] = vmin_0_20
    summary_dict['vmax_20_40'] = vmax_20_40
    summary_dict['vmean_20_40'] = vmean_20_40
    summary_dict['amin_0_50'] = amin_0_50
    summary_dict['amax_0_50'] = amax_0_50
    summary_dict['std_0_50'] = std_0_50
    summary_dict['mad_0_50'] = mad_0_50
    summary_dict['layers'] = ','.join(layer_differences[0])
    for ld in layer_differences:
        key = '%s_%s_px'%(ld[0],ld[1])
        thickness = thickness_dict[key]
        summary_dict[key] = thickness

    indices_to_drop = summary_df[ (summary_df['date']==summary_dict['date']) & (summary_df['time']==summary_dict['time'])].index
    if len(indices_to_drop)>0:
        logging.info('Duplicate item found in summary CSV. Replacing old version with new.')
        
    summary_df.drop(indices_to_drop, inplace=True)
    new_row = pd.Series(summary_dict)

    #summary_df = pd.concat([summary_df,new_df],axis=0)
    summary_df = summary_df.append(new_row,ignore_index=True)
    
    if not style in styles_with_origin:
        add_origin()
        
    plt.xlabel('time (ms)')
    plt.ylabel('velocity (%s)'%velocity_label)
    plt.ylim(rel_plot_clim)
    plt.xlim(tlim_ms)
    #opf.despine()
    leg = plt.legend(frameon=show_legend_frame)
    format_legend(leg)
    savefig('layer_velocity_differences',tag)
        
    make_profile_plot(prof,peak_dict)
    add_to_data_dictionary(folder,'profile',prof)
    plt.axhline(profile_peak_threshold)
    plt.text(0,profile_peak_threshold,'peak threshold',ha='left',va='bottom')
    #opf.despine()
    savefig('peak_profiles',tag)



    
    plt.figure(figsize=[figure_size[0]/2.0,figure_size[1]])
    temp = np.arange(len(prof))
    plt.plot(prof,temp,linewidth=plot_linewidth)
    plt.xticks([])
    plt.yticks([])

    xlim = (np.nanmin(prof)-0.1*np.nanmax(prof),np.nanmax(prof)*2)
    ylim = (np.max(temp),np.min(temp))
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.text(xlim[1]*.95,np.mean(ylim),'depth',ha='right',va='center',rotation=-90)
    #plt.text(np.mean(xlim),ylim[1]*.5,'amplitude',ha='center',va='top')
    
    #plt.gca().set_xlabel('amplitude',labelpad=2)
    #plt.gca().set_ylabel('depth',rotation=-90,labelpad=amplitude_label_padding)
    
    #plt.gca().xaxis.set_label_position('top')
    #plt.gca().yaxis.set_label_position('right')
    
    savefig('peak_profile_vertical',tag)

    

    
    fig,ax = show_bscan(bscan)
    savefig('bscan_amp',tag)
    scalebars(3.0,2.5,ax=ax)
    add_to_data_dictionary(tag,'bscan',bscan)
    
    fig,ax = show_bscan_overlay(bscan,prestim)
    scalebars(3.0,2.5,ax=ax)
    savefig('bscan_pre_stim',tag)
    add_to_data_dictionary(tag,'bscan_prestim',prestim)
    
    fig,ax = show_bscan_overlay(bscan,poststim)
    scalebars(3.0,2.5,ax=ax)
    savefig('bscan_post_stim',tag)
    add_to_data_dictionary(tag,'bscan_poststim',poststim)

    fig,ax = show_bscan(bscan,layer_dict=peak_dict)
    savefig('bscan_amp_layers',tag)
    scalebars(3.0,2.5,ax=ax)

    amp_m = nanmean(ablock[:,:,stimulated_region_start:stimulated_region_end],axis=2).T
    vel_m = nanmean(vblock[:,:,stimulated_region_start:stimulated_region_end],axis=2).T
    show_mscan_overlay(amp_m,vel_m,peak_dict=peak_dict)
    savefig('mscan_overlay',tag)
    add_to_data_dictionary(tag,'mscan_amplitude',amp_m)
    add_to_data_dictionary(tag,'mscan_velocity',vel_m)
    
    plt.close('all')
    
    show_mscan_overlay(amp_m,vel_m,peak_dict=peak_dict,do_overlay=False)
    savefig('mscan_no_overlay',tag)
    
    plt.close('all')

    
summary_df.to_csv(summary_datafile)    

if len(tags)==0:
    sys.exit('Error.')
elif len(tags)==1:
    common_string = tags[0]
else:
    common_string = tags[0]
    for test_tag in tags[1:]:
        m = SequenceMatcher(None,common_string,test_tag).find_longest_match(0,len(common_string),0,len(test_tag))
        common_string = test_tag[m.b:m.b+m.size]
    common_string = common_string.strip('_').strip()

logging.info('Working on average plots.')
plt.figure(figsize=figure_size,dpi=screen_dpi)
for peak_label in peak_labels:
    arr = np.array(layer_dict[peak_label])
    avg = nanmean(arr,axis=0)
    std = np.nanstd(arr,axis=0)
    plt.plot(1000*t_arr,avg,label=peak_label,linewidth=plot_linewidth)
    if plot_error_region:
        plt.fill_between(1000*t_arr,y1=avg+std,y2=avg-std,color=noise_color,alpha=noise_alpha)
    if plot_single_measurements:
        for k in range(arr.shape[0]):
            plt.plot(1000*t_arr,arr[k,:],color=single_color,alpha=single_alpha,linewidth=single_linewidth)
            
    add_to_data_dictionary('all','single_layer_mean_%s'%peak_label,avg)
    add_to_data_dictionary('all','single_layer_std_%s'%peak_label,std)

            
if not style in styles_with_origin:
    add_origin()

plt.xlabel('time (ms)')
plt.ylabel('velocity (%s)'%velocity_label)
plt.ylim(abs_plot_clim)
plt.xlim(tlim_ms)
#opf.despine()
leg = plt.legend(frameon=show_legend_frame)
format_legend(leg)
savefig('layer_velocities','average_%s'%common_string)

plt.figure(figsize=figure_size,dpi=screen_dpi)
for a,b in layer_differences:
    a_arr = np.array(layer_dict[a])
    b_arr = np.array(layer_dict[b])
    a_avg = nanmean(a_arr,axis=0)
    b_avg = nanmean(b_arr,axis=0)
    d = a_avg-b_avg
    d_arr = a_arr-b_arr
    
    std = np.sqrt(np.nanstd(a_arr,axis=0)**2+np.nanstd(b_arr,axis=0)**2)
    if a=='COST' and b=='ISOS':
        layer_label = '$v_{OS}$'
    else:
        layer_label = '%s - %s'%(a,b)
    
    plt.plot(1000*t_arr,d,label=layer_label,color=color_cycle[layer_difference_color_index%len(color_cycle)],linewidth=plot_linewidth)
    if plot_error_region:
        plt.fill_between(1000*t_arr,y1=d+std,y2=d-std,color=noise_color,alpha=noise_alpha)
    if plot_single_measurements:
        for k in range(d_arr.shape[0]):
            plt.plot(1000*t_arr,d_arr[k,:],color=single_color,alpha=single_alpha,linewidth=single_linewidth)

    add_to_data_dictionary('all','layer_difference_mean_%s_%s'%(a,b),d)
    add_to_data_dictionary('all','layer_difference_std_%s_%s'%(a,b),std)
            
    
if not style in styles_with_origin:
    add_origin()

plt.xlabel('time (ms)')
plt.ylabel('velocity (%s)'%velocity_label)
plt.ylim(rel_plot_clim)
plt.xlim(tlim_ms)
#opf.despine()
leg = plt.legend(frameon=show_legend_frame)
format_legend(leg)

savefig('layer_velocity_differences','average_%s'%common_string)


data_dictionary_filename = '%s_data.mat'%metatag
logging.info('Saving plotting data to %s.'%data_dictionary_filename)
sio.savemat(data_dictionary_filename,data_dictionary)


logging.info('Writing report.')
figure_list.sort(key=lambda tup: tup[0])


d = {}
d['bscan_amp_layers'] = 'Amplitude B-scans'
d['bscan_pre'] = 'Velocity (pre stimulus)'
d['bscan_post'] = 'Velocity (post stimulus)'
d['mscan_overlay'] = 'M-scans with velocity overlay'
d['mscan_no_overlay'] = 'M-scans'
d['layer_velocities'] = 'Layer velocity plots'
d['layer_velocity_differences'] = 'Relative layer velocity plots'
d['peak_profiles'] = 'Axial profiles'
d['peak_profile_vertical'] = 'Simplified axial profile'

document_figure_labels = ['bscan_amp_layers','peak_profiles','peak_profile_vertical','bscan_pre','bscan_post','mscan_no_overlay','mscan_overlay','layer_velocities','layer_velocity_differences']

if make_pdf:
    try:
        with open('%s/figures.md'%output_folder,'w') as fid:
            for label in document_figure_labels:
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
        for label in document_figure_labels:
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
                fid.write('<img src=\"%s\" alt=\"%s: %s\" height=\"50%%\">\n'%(outfn,plot_type,file_tag))
                fid.write('</a>\n')
                fid.write('</p>\n')
                fid.write('<p>Fig. %d. %s / %s</p>\n'%(fign,plot_type,file_tag))
                fign+=1
            fid.write('<hr>\n\n')
        fid.write('</body>')
except Exception as e:
    print('html writing error:',e)

try:
    if auto_open_report:
        report_fn = os.path.join(output_folder,'figures.html')
        wb.open(report_fn)
        #os.system('firefox %s/figures.html'%output_folder)
except:
    pass
