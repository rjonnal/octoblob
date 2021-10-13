import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob,os,sys
import scipy.signal as sps
import octoblob.plotting_functions as opf

opf.setup_plots()
color_cycle = opf.get_color_cycle()

###### Configuration parameters #########
phase_bscan_smoothing_sigma = 7
t_offset = 47 # for sigma 1
b_phase_clim = 4.0
m_phase_clim = 3.0
show_stimulated_region = False
dt = 0.0025
# approximate region of stimulation, in fast axis coordinates:
stimulated_region_start = 0
stimulated_region_end = 170
profile_peak_threshold = 4000
peak_labels = ['ISOS','COST','RPE']
layer_differences = [('COST','ISOS'),('RPE','ISOS')]
stim_start = 1
stim_end = 3

###### Plotting parameters ######
mscan_figure_size = (3.75,2.25)
mscan_axes_rect = [0.1,0.15,0.95,0.78]
bscan_figure_size = (7.5,2.5)
bscan_axes_rect = [0.15,0.1,0.8,0.8]
plots_figure_size = (3.75,2.25)
plots_axes_rect = [0.12,0.15,0.83,0.8]
screen_dpi = 100
dB_clim = (50,90)
save_dpi = 300


param_str = '_smoothing_%d'%(phase_bscan_smoothing_sigma)
if show_stimulated_region:
    param_str = param_str + '_show_stimulated_region'

def make_tag(path):
    toks = []
    while True:
        tup = os.path.split(path)
        print(tup[1])
        toks = [tup[1]]+toks
        path = tup[0]
        if len(path)==0:
            break
    return '_'.join(toks)

folder = sys.argv[1].strip('/')
output_folder = 'org_block_figures'

try:
    tag = sys.argv[2]
except:
    tag = make_tag(folder)

os.makedirs(output_folder,exist_ok=True)
def savefig(fn):
    outfn = os.path.join(output_folder,'%s_%s.png'%(fn,tag))
    plt.savefig(outfn,dpi=save_dpi)
    

# RSJ: original settings were -6*clim_factor,6*clim_factor:
# original: slope_clim = (-6.000*clim_factor,6.000*clim_factor)
b_slope_clim = (-b_phase_clim,b_phase_clim)
m_slope_clim = (-m_phase_clim,m_phase_clim)

def com(ascan):
    z = np.arange(len(ascan))
    return np.sum(ascan*z)/np.sum(ascan)

def brm(bscan):
    ascan = np.mean(bscan,axis=1)
    high = np.where(ascan>12000)[0]
    return high[-1]

def zreg(ascan,ref):
    ascan = (ascan-np.nanmean(ascan))/np.nanstd(ascan)
    ref = (ref-np.nanmean(ref))/np.nanstd(ref)
    nxc = np.real(np.fft.ifft(np.fft.fft(ascan)*np.conj(np.fft.fft(ref))))
    p = np.argmax(nxc)
    if p>len(ascan)//2:
        p = p - len(ascan)
    return p
    

def make_mask(im,max_frac=0.1):
    out = np.ones(im.shape)*np.nan
    out[im>max_frac*np.nanmax(im)] = 1.0
    return out

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1.0500

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0


fig_m = plt.figure(figsize=mscan_figure_size,dpi=screen_dpi)
ax_m = fig_m.add_axes(mscan_axes_rect)

fig_b = plt.figure(figsize=bscan_figure_size,dpi=screen_dpi)

fig_profs = plt.figure(figsize=(5,3))

fig_plot = plt.figure(figsize=plots_figure_size,dpi=screen_dpi)
ax_plot = fig_plot.add_axes(plots_axes_rect)

fig_diffplot = plt.figure(figsize=plots_figure_size,dpi=screen_dpi)
ax_diffplot = fig_diffplot.add_axes(plots_axes_rect)

stim_color = [0.0,0.5,0.0]#color_cycle[1]
stim_linestyle = '-'

def show_amp(lin,xlim=None,ylim=None,clim=dB_clim,mode='dB',mscan=True):
    sz,sy = lin.shape
    if xlim is None:
        xlim = [0,sy]
    dt = 0.0025
    t = dt*(np.arange(sy)-t_offset)*1000
    z = np.arange(sz)
    extent = (t[0],t[-1],z[-1],z[0])
    if mode=='linear':
        plt.imshow(lin,cmap='gray',aspect='auto',extent=extent,clim=clim)
    elif mode=='dB':
        plt.imshow(20*np.log10(lin),cmap='gray',aspect='auto',extent=extent,clim=clim)
        
    plt.autoscale(False)
    plt.yticks([])
    x = xlim[1]-10
    
    plt.plot([x,x],[10,30],'y-',linewidth=3)
    if not mscan:
        plt.plot([x-50/3.0,x],[10,10],'y-',linewidth=3)
        
    plt.xlim(xlim)
    plt.ylim(ylim)

def show_phase(masked_phase,xlim=None,ylim=None,arrows=[],mscan=True):
    sz,sy = masked_phase.shape
    dt = 0.0025
    t = dt*(np.arange(sy)-48.5)*1000

    z = np.arange(sz)
    if mscan:
        clim = m_slope_clim
    else:
        clim = b_slope_clim
    extent = (t[0],t[-1],z[-1],z[0])
    plt.imshow(masked_phase,cmap='jet',clim=clim,aspect='auto',alpha=0.5,extent=extent)
    plt.xlim(xlim)
    plt.ylim(ylim)
    for ah,ac in arrows:
        dx = -20
        dy = 0
        x = -dx+10
        y = ah-dy
        a = plt.arrow(x,y,dx,dy,head_width=6,width=3,length_includes_head=True)
        a.set_edgecolor('k')
        a.set_facecolor(ac)
        
def mlayer(amp,phase,idx,averaging_half_height=0):
    srad = 1
    sz,sy = amp.shape
    phase_sheet = np.ones((2*averaging_half_height+1,sy))
    idx_vec = np.array(idx_vec) + idx
    for row in range(-averaging_half_height,averaging_half_height+1):
        for y in range(sy):
            phase_sheet[row+averaging_half_height,y] = phase[idx_vec[y]+row,y]

    return phase_sheet.mean(axis=0)

trial_linestyles = ['--',':']

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

def oversample_and_align(m,factor=5):
    sz,sy = m.shape
    new_sz = sz*5
    def oversample(aline):
        return np.abs(np.fft.ifft(np.fft.fftshift(np.fft.fft(aline)),n=new_sz))
    
    out = np.zeros((new_sz,sy))

    elm_idx = []
    for y in range(sy):
        aline = oversample(m[:,y])
        aline[np.where(aline<2000)] = 0
        left = aline[:-2]
        center = aline[1:-1]
        right = aline[2:]
        elm = np.where(np.logical_and(center>left,center>right))[0][0]
        elm_idx.append(elm)

    elm_idx = np.array(elm_idx)
    elm_idx = elm_idx-elm_idx.min()
    for y in range(sy):
        aline = oversample(m[:,y])
        out[:,y] = np.roll(aline,-elm_idx[y])

    return out


flist = sorted(glob.glob(os.path.join(folder,'phase_ramp*.npy')))

z_pos = []

mscan_xlim = (-100,75)

def text(ax,x,y,s,ha='right',va='center',color='w'):
    h = 20
    w = 6.5*len(s)
    #rect = patches.Rectangle((x, y-h/2), w, h, linewidth=0, edgecolor='r', facecolor=color,alpha=0.5)
    #ax.add_patch(rect)
    ax.text(x-2,y,s,ha=ha,va=va,color='k',fontsize=6)

for idx,f in enumerate(flist):

    temp = np.load(f)
    amp = np.real(temp)


    phase_slope = np.imag(temp)


    #########################################
    # IMPORTANT: RSJ: I'm reversing the sign of the phase slope, just to make the data conform
    # to expectations about contraction/elongation; make sure to prove that this is correct before
    # publication.
    #########################################
    phase_slope = -phase_slope

    phase_slope = phase_to_nm(phase_slope)

    phase_kernel = make_kernel(phase_bscan_smoothing_sigma,phase_bscan_smoothing_sigma)
    phase_slope = sps.convolve2d(phase_slope,phase_kernel,mode='same')

    if idx==0:
        sz,sx = amp.shape
        sy = len(flist)
        amp_block = np.zeros((sy,sz,sx))
        phase_slope_block = np.zeros((sy,sz,sx))
        ref = np.nanmean(amp,axis=1)

    amp_block[idx,:,:] = amp
    phase_slope_block[idx,:,:] = phase_slope

    z_pos.append(zreg(np.nanmean(amp,axis=1),ref))


t = dt*(np.arange(sy)-t_offset)*1000

dz = 2.5
z = np.arange(sz)
stim_idx = np.argmin(np.abs(t))

def linereg(a,b,mask=None):
    if mask is None:
        mask = np.ones(a.shape)

    aa = (a-a.mean())/a.std()
    bb = (b-b.mean())/b.std()
    xc = np.real(np.fft.ifft(np.fft.fft(aa*mask)*np.conj(np.fft.fft(bb*mask))))
    p = np.argmax(xc)
    if p>len(a)//2:
        p = p - len(a)
    return a,np.roll(b,p)


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


# plot profile, threshold, and labels
peri_stim_amp = np.nanmean(amp_block[stim_idx-20:stim_idx+20,:,:],axis=0)
prof = np.nanmean(peri_stim_amp,axis=1)
peak_dict = get_peak_dict(prof)
mask = make_mask(peri_stim_amp)
pre_stim_phase = np.nanmean(phase_slope_block[stim_idx-10:stim_idx-1,:,:],axis=0)*mask
post_stim_phase = np.nanmean(phase_slope_block[stim_idx+stim_start:stim_idx+stim_end,:,:],axis=0)*mask

def add_labels(ax,xlim=None):
    if xlim is None:
        xlim = plt.gca().get_xlim()

    for k in peak_dict.keys():
        if k in peak_labels:
            text(ax,xlim[0],peak_dict[k],k)

plt.figure(fig_profs.number)
plt.plot(prof)
for kidx,key in enumerate(peak_dict.keys()):
    c = color_cycle[kidx%len(color_cycle)]
    idx = peak_dict[key]
    plt.axvline(idx,color=c)
    plt.text(idx,prof[idx],key,ha='right',va='bottom',rotation=90,color=c,fontsize=6)
plt.yticks([])
plt.ylabel('OCT amplitude')
opf.despine()
savefig('profile_peaks')

mask = make_mask(peri_stim_amp)
phase_slope_block[np.where(phase_slope_block==0)] = np.nan

title_string = r'dz/dt ($\mu$m/s)'
plt.figure(fig_b.number)
plt.axes([0.05,0.03,0.32,0.85])
show_amp(peri_stim_amp)
add_labels(plt.gca())
plt.colorbar()
plt.ylim((135,5))
plt.xticks([])
plt.title('B-scan (dB)')

plt.axes([0.39,0.03,0.32,0.85])
show_amp(peri_stim_amp)
show_phase(pre_stim_phase,mscan=False)
if show_stimulated_region:
    plt.gca().add_patch(stim_rect1)
plt.colorbar()
plt.ylim((135,5))
plt.xticks([])
plt.title('pre-stimulus %s'%title_string)

plt.axes([0.73,0.03,0.26,0.85])
show_amp(peri_stim_amp)
show_phase(post_stim_phase,mscan=False)
if show_stimulated_region:
    plt.gca().add_patch(stim_rect2)
#plt.colorbar()
plt.ylim((135,5))
plt.xticks([])
plt.title('post-stimulus %s'%title_string)
savefig('bscans')



############### MSCANS ################################

amp_m = np.nanmean(amp_block[:,:,stimulated_region_start:stimulated_region_end],axis=2).T
phase_m = np.nanmean(phase_slope_block[:,:,stimulated_region_start:stimulated_region_end],axis=2).T
phase_m[:,:15] = np.nan

#for y in range(sy):
mask = make_mask(amp_m)

extent = (t[0],t[-1],z[-1],z[0])

zero_idx = np.argmin(np.abs(t))
plt.figure(fig_m.number)
show_amp(amp_m,mscan_xlim)
show_phase(mask*phase_m,mscan_xlim)

plt.axvline(0.0,color=stim_color,linestyle=stim_linestyle)
plt.xlim(mscan_xlim)
plt.colorbar()
plt.xlabel('time (ms)')
plt.title(r'dz/dt ($\mu$m/s)')

plt.yticks([])
ax = plt.gca()
add_labels(ax)

savefig('mscan')

############### PLOTS ################################


plt.figure(fig_plot.number)
for idx,peak_label in enumerate(peak_labels):
    if peak_label=='':
        continue
    #layer_phase = mlayer(amp_m,phase_m,peak_dict[peak_label])
    layer_phase = phase_m[peak_dict[peak_label],:]
    layer_phase = layer_phase - np.nanmean(layer_phase[:stim_idx])
    plt.plot(t,layer_phase,color=color_cycle[idx],label=peak_label)
    
plt.xlim(mscan_xlim)
plt.ylabel(r'dz/dt ($\mu$m/s)')
plt.xlabel('time (ms)')
plt.axvline(0.0,color=stim_color,linestyle=stim_linestyle)
plt.legend()
opf.despine()
savefig('phase_slope_by_layer')



plt.figure(fig_diffplot.number)
for idx,pair in enumerate(layer_differences):
    key2,key1 = pair
    idx1 = peak_dict[key1]
    idx2 = peak_dict[key2]
    layer_phase_1 = phase_m[idx1,:]
    layer_phase_2 = phase_m[idx2,:]
    dphase = layer_phase_2-layer_phase_1
    plt.plot(t,dphase,color=color_cycle[idx],label='%s-%s'%(key2,key1))
    
plt.axhline(0,color='k')
plt.xlim(mscan_xlim)
plt.ylabel(r'dz/dt ($\mu$m/s)')
plt.xlabel('time (ms)')
plt.axvline(0.0,color=stim_color,linestyle=stim_linestyle)
plt.legend()
opf.despine()
savefig('phase_slope_layer_differences')

plt.show()
