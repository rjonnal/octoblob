import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
import glob
import os,sys
import octoblob as blob
from octoblob import config_reader
import scipy.ndimage as spn
import scipy
print(scipy.__version__)

from octoblob.registration_tools import rigid_register

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

PHASE_CLIM = (0,2*np.pi)

# setup parameters
unp_filename = sys.argv[1]
flags = [t.lower() for t in sys.argv[2:]]

cfg = config_reader.get_configuration(unp_filename.replace('.unp','.xml'))

n_vol = cfg['n_vol']
n_slow = cfg['n_slow']
n_repeats = cfg['n_bm_scans']
n_fast = cfg['n_fast']
n_depth = cfg['n_depth']

# some conversions to comply with old conventions:
n_slow = n_slow//n_repeats
n_fast = n_fast*n_repeats

# Frame rate is derived from Axsun line rate and n_fast,
# which is the number of A-scans in a B-scan.
# Remember that the B-scan width may not be the same as
# n_fast, since B-scans are cropped during processing.
# The same is true for n_depth.
dt = n_fast/1e5 
logging.info('B-scan interval: %0.1f ms; B-scan rate: %0.1f Hz'%(dt*1000,1/dt))

folder_name = '%s_bscans'%unp_filename.replace('.unp','')
filter_string = os.path.join(folder_name,'complex*.npy')
flist = sorted(glob.glob(filter_string))
n_files = len(flist)


output_directory = unp_filename.replace('.unp','')+'_phase_ramps'
os.makedirs(output_directory,exist_ok=True)


logging.info('Searched %s; found %d files.'%(filter_string,n_files))

def oversample(im,factor):
    if all([x==1 for x in factor]):
        return im
    else:
        im_f = np.fft.ifft2(im)
        sz,sx = im.shape
        new_s = (int(sz*factor[0]),int(sx*factor[1]))
        oversampled = np.fft.fft2(im_f,s=new_s)
    return oversampled

def shift_image(im,shift,fill_value=np.nan):
    r = np.real(im)
    c = np.imag(im)
    r = spn.shift(r,shift,cval=fill_value)
    c = spn.shift(c,shift,cval=fill_value)
    return r+c*1j

def show_both(a,b):
    plt.subplot(1,2,1)
    plt.imshow(np.abs(a))
    plt.subplot(1,2,2)
    plt.imshow(np.abs(b))
    plt.show()

def strip_align_pair(a,b,n_strips,diagnostics=False):
    sz,sx = a.shape
    strip_width = sx//n_strips

    x0 = 0
    x1 = strip_width

    new_b = []
    while x0<sx:
        a_strip = a[:,x0:x1]
        b_strip = b[:,x0:x1]
        dx,dz,xc = rigid_register(a_strip,b_strip)
        x0+=strip_width
        x1+=strip_width
        if dx or dz:
            b_strip = shift_image(b_strip,(dz,dx))
        new_b.append(b_strip)
    return np.hstack(new_b)

block_size = 5
block_start = 0
oversample_factor = (1,1)

oversampled_dict = {}

def get_oversampled_file(fn,factor,odict):
    try:
        ov = odict[fn]
    except:
        ov = np.load(fn)
        ov = oversample(ov,factor)
        odict[fn] = ov
    return ov
        
while block_start+block_size<=n_files:
    files = flist[block_start:block_start+block_size]
    ref = get_oversampled_file(files[0],oversample_factor,oversampled_dict)
    
    logging.info('strip_align_pair: block %d'%block_start)

    block = [ref]
    for tar_idx in range(1,block_size):
        logging.info('strip_align_pair: images %d and %d'%(block_start,block_start+tar_idx))
        tar = get_oversampled_file(files[tar_idx],oversample_factor,oversampled_dict)
        registered = strip_align_pair(ref,tar,10)
        block.append(registered)

    block = np.array(block)
    block = np.transpose(block,(1,2,0))

    average_bscan = np.mean(np.abs(block),axis=2)
    mask = np.zeros(average_bscan.shape)
    threshold = np.max(average_bscan)*0.1
    mask[average_bscan>threshold] = 1
    
    block_phase = np.angle(block)

    corrected_block_phase = blob.bulk_motion_correct(block_phase,mask,diagnostics=False)

    corrected_block_phase = np.transpose(np.transpose(corrected_block_phase,(2,0,1))*mask,(1,2,0))
    
    corrected_block_phase = np.unwrap(corrected_block_phase,axis=2)
    corrected_block_phase = corrected_block_phase%(2*np.pi)

    block_dphase = np.diff(corrected_block_phase,axis=2)
    phase_slope = np.mean(block_dphase,axis=2)
    phase_slope[np.where(mask==0)] = np.nan
    
    plt.clf()
    plt.imshow(20*np.log10(average_bscan),cmap='gray',aspect='auto')
    plt.imshow(phase_slope,cmap='jet',alpha=0.33,aspect='auto')
    plt.colorbar()
    plt.title(r'phase ramp (d$\theta$/dt), frames %02d-%02d (t = %0.3f, %0.3f)'%(block_start,block_start+block_size-1,dt*block_start,dt*(block_start+block_size-1)))
    outfn = os.path.join(output_directory,'phase_ramp_frames_%02d-%02d.png'%(block_start,block_start+block_size-1))
    plt.savefig(outfn,dpi=100)
    plt.pause(.0001)


    
    block_start+=1
