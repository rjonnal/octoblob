import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob,os,sys
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', family='serif')
plt.rc('font', serif=['Times New Roman'])
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


folder = sys.argv[1]
phase_ramp_clim = (-80,80)

flist = sorted(glob.glob(os.path.join(folder,'phase_ramp*.npy')))

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
    

def make_mask(im,max_frac=0.03):
    out = np.ones(im.shape)*np.nan
    out[im>max_frac*np.nanmax(im)] = 1.0
    return out

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0



z_pos = []

for idx,f in enumerate(flist):

    temp = np.load(f)
    amp = np.real(temp)
    dl = np.imag(temp)

    dl[np.where(dl==0)] = np.nan
    
    if idx==0:
        sz,sx = amp.shape
        sy = len(flist)
        amp_block = np.zeros((sy,sz,sx))
        dl_block = np.zeros((sy,sz,sx))
        ref = np.nanmean(amp,axis=1)

    amp_block[idx,:,:] = amp
    dl_block[idx,:,:] = dl

    z_pos.append(zreg(np.nanmean(amp,axis=1),ref))

if False:
    plt.figure()
    plt.imshow(np.nanmean(amp_block,axis=2).T,cmap='gray')

    z_pos = z_pos-np.mean(z_pos)
    z_pos = np.round(-z_pos).astype(np.int)

    for y in range(sy):
        amp_block[y,:,:] = np.roll(amp_block[y,:,:],z_pos[y],axis=0)
        dl_block[y,:,:] = np.roll(dl_block[y,:,:],z_pos[y],axis=0)
    

    plt.figure()
    plt.imshow(np.nanmean(amp_block,axis=2).T,cmap='gray')
    
    plt.show()
        
stimulus_border = 130

amp_m = np.nanmean(amp_block[:,:,:stimulus_border],axis=2).T
dl_m = np.nanmean(dl_block[:,:,:stimulus_border],axis=2).T
dl_m[:,:20] = np.nan

#for y in range(sy):
mask = make_mask(amp_m,0.24)

dt = 0.0025
t = dt*(np.arange(sy)-48.5)*1000

dz = 2.5
z = dz*np.arange(sz)

extent = (t[0],t[-1],z[-1],z[0])

nm_clim = (phase_to_nm(phase_ramp_clim[0]),phase_to_nm(phase_ramp_clim[1]))

xlim = (-100,75)



def z_extent_to_idx(z):
    e = np.linspace(extent[3],extent[2],sz)
    return np.argmin(np.abs(e-z))


outdir = '/home/rjonnal/figsrc/conventional_org_pilot_data'
os.makedirs(outdir,exist_ok=True)

nm = phase_to_nm(dl_m)
#plt.figure(figsize=(3,3),dpi=100)
#plt.imshow(amp_m,cmap='gray',aspect='auto')
#plt.imshow(mask*nm,cmap='jet',clim=nm_clim,aspect='auto',alpha=0.5)

if folder.find('16_58_12')>-1:
    cost_idx = 104
    z_offset = 4       
elif folder.find('16_53_25')>-1:
    cost_idx = 100
    z_offset = 0
    
elm_idx = 213 + z_offset
isos_idx = 235 + z_offset
cost_idx = 249 + z_offset
rpe_idx = 263 + z_offset
ilm_idx = 84 + z_offset
ipl_idx = 126 + z_offset
opl_idx = 160 + z_offset
ch_idx = 286 + z_offset

def text(ax,x,y,s,ha='left',va='center',color='w'):
    h = 20
    w = 5.5*len(s)
    rect = patches.Rectangle((x, y-h/2), w, h, linewidth=0, edgecolor='r', facecolor=color,alpha=0.5)
    ax.add_patch(rect)
    ax.text(x,y,s,ha=ha,va=va,color='k')

plt.figure(figsize=(4,4))
plt.imshow(amp_m,cmap='gray',aspect='auto',extent=extent)
plt.imshow(mask*dl_m,cmap='jet',clim=phase_ramp_clim,aspect='auto',alpha=0.5,extent=extent)
#plt.imshow(mask*phase_to_nm(dl_m),cmap='jet',clim=nm_clim,aspect='auto',alpha=0.5,extent=extent)

plt.axvline(0.0,color='g')
plt.xlim(xlim)
plt.colorbar()
plt.xlabel('time (ms)')
plt.title(r'd$\theta$/dt (rad/s)')
plt.ylabel('z ($\mu m$)')
ax = plt.gca()
text(ax,xlim[0]+2,elm_idx,'ELM',ha='left',va='center')
text(ax,xlim[0]+2,isos_idx,'ISOS',ha='left',va='center')
text(ax,xlim[0]+2,cost_idx,'COST',ha='left',va='center')
text(ax,xlim[0]+2,rpe_idx,'RPE',ha='left',va='center')
text(ax,xlim[0]+2,ilm_idx,'ILM',ha='left',va='center')
text(ax,xlim[0]+2,ipl_idx,'IPL',ha='left',va='center')
text(ax,xlim[0]+2,opl_idx,'OPL',ha='left',va='center')
text(ax,xlim[0]+2,ch_idx,'Ch',ha='left',va='center')

fn = os.path.join(outdir,'%s_org_mscan.png'%folder.split('/')[-4])
print(fn)
plt.savefig(fn,dpi=300)

rad=2
cost_phase = phase_to_nm(dl_m)[cost_idx-rad:cost_idx+rad+1,:]
cost_phase = np.mean(cost_phase,axis=0)


plt.figure(figsize=(3,3),dpi=100)
plt.plot(t,cost_phase)


plt.show()
