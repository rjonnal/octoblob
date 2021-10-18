import imageio,glob,os,sys
import numpy as np
from matplotlib import pyplot as plt

in_folder = sys.argv[1]
out_folder = sys.argv[2]

os.makedirs(out_folder,exist_ok=True)

try:
    mode = sys.argv[3]
except:
    mode = 'dB'

try:
    abs_clim = float(sys.argv[4])
except:
    abs_clim = 7
    
write = 'write' in sys.argv[1:]
show = 'show' in sys.argv[1:]

flist = glob.glob(os.path.join(in_folder,'*.npy'))

flist.sort()

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1.0500

ms = 0.0

for idx,f in enumerate(flist):
    print('File %d of %d.'%(idx+1,len(flist)))
    im = np.load(f)
    plt.cla()
    if mode=='dB':
        png = 20*np.log10(np.abs(im))
        plt.imshow(png,cmap='gray',clim=(40,80))
    if mode=='phase_ramp':
        amp = np.real(im)
        phase_slope = np.imag(im)
        phase_slope = phase_to_nm(phase_slope)
        phase_slope[phase_slope==0] = np.nan
        plt.imshow(20*np.log10(np.abs(amp)),cmap='gray',clim=(40,80))
        plt.imshow(phase_slope,cmap='jet',alpha=0.5,clim=(-abs_clim,abs_clim))
        plt.title('t=%0.1f ms'%ms)

    fpng = os.path.split(f)[1].replace('.npy','')+'.png'
    outfn = os.path.join(out_folder,fpng)
    if write:
        plt.savefig(outfn)
    ms = ms + 2.5
    
    if show:
        plt.pause(.001)

