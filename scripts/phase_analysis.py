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
from octoblob.ticktock import tick,tock

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

histogram_threshold_fraction = 0.03
signal_threshold_fraction = 0.05
unfiltered_min = -5000
unfiltered_max = 5000
filtered_min = -2500
filtered_max = 2500


# the duration over which we assume the retina is stationary (in seconds)
stationary_duration = 0.01
testing = False

phase_clim = (0,2*np.pi)

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

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
block_size = int(round(stationary_duration/dt))
logging.info('Setting block size to %d, assuming retina is stationary for %d ms.'%(block_size,stationary_duration*1000))

folder_name = '%s_bscans'%unp_filename.replace('.unp','')
filter_string = os.path.join(folder_name,'complex*.npy')
flist = sorted(glob.glob(filter_string))
n_files = len(flist)


npy_output_directory = unp_filename.replace('.unp','')+'_phase_ramps'
png_output_directory = unp_filename.replace('.unp','')+'_phase_ramps'

os.makedirs(npy_output_directory,exist_ok=True)
os.makedirs(png_output_directory,exist_ok=True)


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


plt.figure(figsize=(10,4),dpi=100)


### Figure out what the histogram max is, just for visualization:
t0 = tick()
temp_block_start = 0
hist_max = 0
while temp_block_start+block_size<=n_files:
    files = flist[temp_block_start:temp_block_start+block_size]
    ref = get_oversampled_file(files[0],oversample_factor,oversampled_dict)
    
    block = [ref]
    for tar_idx in range(1,block_size):
        tar = get_oversampled_file(files[tar_idx],oversample_factor,oversampled_dict)
        block.append(tar)

    block = np.array(block)
    average_bscan = np.nanmean(np.abs(block),axis=0)
    
    signal_mask = np.zeros(average_bscan.shape)

    # there may be nans, so use nanmax
    signal_threshold = np.nanmax(average_bscan)*signal_threshold_fraction
    
    signal_mask[average_bscan>signal_threshold] = 1
    hist_max_temp = np.sum(signal_mask)/8.0
    if hist_max_temp>hist_max:
        hist_max = hist_max_temp
    temp_block_start = temp_block_start + 5
    

### Begin processing blocks:
t0 = tick()
block_start = 0
while block_start+block_size<=n_files:
    
    files = flist[block_start:block_start+block_size]
    ref = get_oversampled_file(files[0],oversample_factor,oversampled_dict)
    
    logging.info('processing block %d'%block_start)

    block = [ref]
    for tar_idx in range(1,block_size):
        tar = get_oversampled_file(files[tar_idx],oversample_factor,oversampled_dict)
        registered = strip_align_pair(ref,tar,10)
        block.append(registered)

    block = np.array(block)
    # in test set, depth is 200, fast is 100, and block size is 5
    # block shape right now, after np.array, is 5, 200, 100
    # let's keep this as our convention, following the volume averaging
    # convention

    if testing:
        block[2,-80:-70,20:80] = np.nan
        plt.imshow(np.abs(block[2,:,:]))
        plt.title('Testing B-scan (with nan region).')
        plt.show()
        
    average_bscan = np.nanmean(np.abs(block),axis=0)
    
    histogram_mask = np.zeros(average_bscan.shape)
    signal_mask = np.zeros(average_bscan.shape)

    # there may be nans, so use nanmax
    histogram_threshold = np.nanmax(average_bscan)*histogram_threshold_fraction
    signal_threshold = np.nanmax(average_bscan)*signal_threshold_fraction
    
    histogram_mask[average_bscan>histogram_threshold] = 1
    signal_mask[average_bscan>signal_threshold] = 1
    
    block_phase = np.angle(block)

    transposed = np.transpose(block_phase,(1,2,0))
    corrected_block_phase = blob.bulk_motion_correct(transposed,histogram_mask,diagnostics=False)
    corrected_block_phase = np.transpose(corrected_block_phase,(2,0,1))


    # the original version is below, and gets completely broken by nans:
    #corrected_block_phase = corrected_block_phase%(2*np.pi)
    #block_dphase = np.diff(corrected_block_phase,axis=2)
    #phase_slope = np.mean(block_dphase,axis=2)
    #phase_slope[np.where(signal_mask==0)] = np.nan

    # nans mess this up:
    #corrected_block_phase = np.unwrap(corrected_block_phase,axis=0)

    # The following approach doesn't work because masking turns corrected_block_phase into a 1D array;
    # the only way I can think of solving this problem is by for loop through the signal_mask pixels.
    # Keep thinking of a way to vectorize.
    #corrected_block_phase[~np.isnan(corrected_block_phase)] = np.unwrap(corrected_block_phase[~np.isnan(corrected_block_phase)],axis=0)

    valid_depth,valid_fast = np.where(signal_mask==1)
    
    t_vec = np.arange(block_size)*dt

    phase_slope_image = np.ones(signal_mask.shape)*np.nan
    nm_slope_image = np.ones(signal_mask.shape)*np.nan
    nm_slope_image_filtered = np.ones(signal_mask.shape)*np.nan
    high_vals = 0
    high_vals2 = 0
    for depth,fast in zip(valid_depth,valid_fast):
        theta = corrected_block_phase[:,depth,fast]
        theta_unwrapped = np.ones(theta.shape)*np.nan
        nan_mask = ~np.isnan(theta)
        theta_unwrapped[nan_mask] = np.unwrap(theta[nan_mask])
        changed = any([a!=b for a,b in zip(theta[nan_mask],theta_unwrapped[nan_mask])])
        if any(np.isnan(theta)) and changed:
            logging.info('nan values found in phase ramp; repaired:')
            logging.info('%s->%s'%(theta,theta_unwrapped))
            
        dt_vec = np.diff(t_vec[nan_mask])
        dtheta_vec = np.diff(theta[nan_mask])

        if True:
            for k in range(len(dtheta_vec)):
                v = dtheta_vec[k]
                while np.abs(v-2*np.pi)<np.abs(v):
                    v = v - 2*np.pi
                while np.abs(v+2*np.pi)<np.abs(v):
                    v = v + 2*np.pi
                dtheta_vec[k] = v

        nm_vec = phase_to_nm(dtheta_vec)
        nm_slope = np.mean(nm_vec/dt_vec)
        phase_slope = np.mean(dtheta_vec/dt_vec)
        
        phase_slope_image[depth,fast] = phase_slope
        nm_slope_image[depth,fast] = nm_slope
        
        if filtered_min<nm_slope<filtered_max:
            nm_slope_image_filtered[depth,fast] = nm_slope
        
        if np.abs(phase_slope)>1000:
            high_vals+=1
            
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(20*np.log10(average_bscan),cmap='gray',aspect='auto')
    plt.imshow(nm_slope_image,cmap='jet',alpha=0.33,aspect='auto',clim=[-5000,5000])
    plt.colorbar()
    plt.title('phase ramp (dL/dt)(nm)')
    plt.suptitle('frames %02d-%02d (t = %0.3f, %0.3f)'%(block_start,block_start+block_size-1,dt*block_start,dt*(block_start+block_size-1)))
    plt.subplot(1,3,2)
    plt.imshow(20*np.log10(average_bscan),cmap='gray',aspect='auto')
    plt.imshow(nm_slope_image_filtered,cmap='jet',alpha=0.9,aspect='auto',clim=[unfiltered_min,unfiltered_max])
    plt.colorbar()
    plt.title('phase ramp (dL/dt)(nm) (filtered)')

    
    plt.subplot(1,3,3)
    plt.hist(nm_slope_image[~np.isnan(nm_slope_image)],bins=np.arange(unfiltered_min,unfiltered_max,500))
    plt.xlabel('dL/dt')
    plt.ylabel('count')
    plt.title('histogram')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.ylim((0,hist_max))
    plt.axvspan(filtered_min,filtered_max,alpha=0.25)
    
    png_outfn = os.path.join(png_output_directory,'phase_ramp_frames_%02d-%02d.png'%(block_start,block_start+block_size-1))
    npy_outfn = os.path.join(npy_output_directory,'phase_ramp_frames_%02d-%02d.npy'%(block_start,block_start+block_size-1))
    plt.savefig(png_outfn,dpi=100)
    np.save(npy_outfn,phase_slope_image)
    plt.pause(.0001)


    
    block_start+=1
    ttemp = tock(t0)
    bbs = block_start/ttemp
    time_remaining = (n_files-block_start-block_size)/bbs
    logging.info('processed %d blocks in %0.1f s (%0.2f blocks per second)'%(block_start,ttemp,bbs))
    logging.info('time remaining: %d s'%time_remaining)
    if testing:
        logging.info('Testing set to True, so quitting after first iteration.')
        sys.exit()
