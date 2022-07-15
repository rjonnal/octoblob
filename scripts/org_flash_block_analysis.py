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
        logging.FileHandler("octoblob.log"),
        logging.StreamHandler()
    ]
)

histogram_threshold_fraction = 0.03
signal_threshold_fraction = 0.04
theta_bin_min = -80
theta_bin_max = 80
theta_bin_step = (theta_bin_max-theta_bin_min)/20.0

# the duration over which we assume the retina is stationary (in seconds)
stationary_duration_default = 0.0075
testing = False
do_strip_registration = False

phase_clim = (0,2*np.pi)

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0

print(nm_to_phase(5000)/(np.pi))

def usage():
    print('Usage:')
    print('\t python org_flash_block_analysis.py path/to/phase_ramp_npy start_index end_index')

# setup parameters
try:
    bscan_folder = sys.argv[1]
except IndexError:
    usage()
    sys.exit()

try:
    block_start = int(sys.argv[2])
    n_files = int(sys.argv[3])
except IndexError:
    usage()
    sys.exit()

try:
    stationary_duration = float(sys.argv[4])
except:
    stationary_duration = stationary_duration_default

try:
    t_stim_start = float(sys.argv[5])
    t_stim_end = float(sys.argv[6])
except:
    t_stim_start = np.inf
    t_stim_end = -np.inf

try:
    x_stim_start = int(sys.argv[7])
    x_stim_end = int(sys.argv[8])
except:
    x_stim_start = np.inf
    x_stim_end = -np.inf
    
flags = [t.lower() for t in sys.argv[2:]]
show = 'show' in flags
diagnostics = 'diagnostics' in flags

if diagnostics:
    diagnostics_directory = bscan_folder.replace('_bscans','')+'_diagnostics'
    os.makedirs(diagnostics_directory,exist_ok=True)

found_xml = False
temp = bscan_folder

while not found_xml:
    xml_filename = temp+'.xml'
    found_xml = os.path.exists(xml_filename)
    if found_xml:
        break
    temp = temp[:-1]

cfg = config_reader.get_configuration(xml_filename)

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
logging.info('Setting block size to %d, assuming retina is stationary for %0.1f ms.'%(block_size,stationary_duration*1000))

filter_string = os.path.join(bscan_folder,'complex*.npy')
flist = sorted(glob.glob(filter_string))

if block_start is None:
    block_start = 0

if n_files is None:
    n_files = len(flist)

logging.info('Searched %s; found %d files.'%(filter_string,len(flist)))
logging.info('Processing blocks %d through %d.'%(block_start,n_files-block_size))
    
blocks_processed = 0

npy_output_directory = os.path.join(bscan_folder,'phase_ramps_%03dms_npy'%(stationary_duration*1000))
png_output_directory = os.path.join(bscan_folder,'phase_ramps_%03dms_png'%(stationary_duration*1000))

os.makedirs(npy_output_directory,exist_ok=True)
os.makedirs(png_output_directory,exist_ok=True)


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


if show:
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
while block_start+block_size<=n_files:
    
    files = flist[block_start:block_start+block_size]
    ref = get_oversampled_file(files[0],oversample_factor,oversampled_dict)
    
    logging.info('processing block %d'%block_start)

    block = [ref]
    for tar_idx in range(1,block_size):
        tar = get_oversampled_file(files[tar_idx],oversample_factor,oversampled_dict)
        if do_strip_registration:
            tar = strip_align_pair(ref,tar,5)
        block.append(tar)

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

    if diagnostics:
        diagnostics_flag = (diagnostics_directory,block_start)
    else:
        diagnostics_flag = False
    
    corrected_block_phase = blob.bulk_motion_correct(transposed,histogram_mask,diagnostics=diagnostics_flag)
    if diagnostics:
        plt.show()
        plt.close('all')
        
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
    
    t_vec = np.arange(block_start,block_start+block_size)*dt
    current_time = np.mean(t_vec)
    stimulus_on = current_time>=t_stim_start and current_time<=t_stim_end

    phase_slope_image = np.ones(signal_mask.shape)*np.nan
    phase_slope_image_all = np.zeros(signal_mask.shape)
    nm_slope_image = np.ones(signal_mask.shape)*np.nan
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
        if not np.isnan(phase_slope):
            phase_slope_image_all[depth,fast] = phase_slope
            
        nm_slope_image[depth,fast] = nm_slope
        
        if np.abs(phase_slope)>1000:
            high_vals+=1

    if show:
        plt.clf()
        plt.subplot(1,2,1)
        plt.cla()
        plt.imshow(20*np.log10(average_bscan),cmap='gray',aspect='auto')
        plt.imshow(phase_slope_image,cmap='jet',alpha=0.75,aspect='auto',clim=[theta_bin_min,theta_bin_max])
        if stimulus_on:
            for k in range(x_stim_start,x_stim_end,20):
                plt.plot(k,10,'gv',markersize=12)
        plt.colorbar()
        plt.title(r'phase ramp ($\theta$/dt)(rad)')
        plt.suptitle('frames %02d-%02d (t = %0.3f, %0.3f)'%(block_start,block_start+block_size-1,dt*block_start,dt*(block_start+block_size-1)))
        
        plt.subplot(1,2,2)
        plt.cla()
        plt.hist(phase_slope_image[~np.isnan(phase_slope_image)],bins=np.arange(theta_bin_min,theta_bin_max,theta_bin_step))
        plt.xlabel(r'd$\theta$/dt')
        plt.ylabel('count')
        plt.title('histogram')
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position('right')
        plt.ylim((0,hist_max))
        #plt.axvspan(filtered_min,filtered_max,alpha=0.25)
        png_outfn = os.path.join(png_output_directory,'phase_ramp_frames_%05d-%05d.png'%(block_start,block_start+block_size-1))
        plt.savefig(png_outfn,dpi=100)
    
    npy_outfn = os.path.join(npy_output_directory,'phase_ramp_frames_%05d-%05d.npy'%(block_start,block_start+block_size-1))
    np.save(npy_outfn,average_bscan+1j*phase_slope_image_all)

    if show:
        plt.pause(.0001)


    
    block_start+=1
    blocks_processed+=1
    ttemp = tock(t0)
    bps = float(blocks_processed)/ttemp
    time_remaining = (n_files-block_start-block_size)/bps
    logging.info('processed %d blocks in %0.1f s (%0.2f blocks per second)'%(blocks_processed,ttemp,bps))
    logging.info('time remaining: %d s'%time_remaining)
    if testing:
        logging.info('Testing set to True, so quitting after first iteration.')
        sys.exit()
