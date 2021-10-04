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

def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

# setup parameters
bscan_folder = sys.argv[1]

output_directory = os.path.join(bscan_folder,'isos_normalized_phase')
os.makedirs(output_directory,exist_ok=True)

try:
    file_start = int(sys.argv[2])
    file_end = int(sys.argv[3])
except:
    file_start = 0
    file_end = 20

n_files = file_end-file_start

flags = [t.lower() for t in sys.argv[2:]]
show = 'show' in flags

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

t_vec = np.arange(file_start,file_end)*dt

filter_string = os.path.join(bscan_folder,'complex*.npy')
flist = sorted(glob.glob(filter_string))[file_start:file_start+n_files]

block = []
for f in flist:
    block.append(np.abs(np.load(f)))

average_bscan = np.mean(block,axis=0)
mask = np.zeros(average_bscan.shape)
mask[average_bscan>np.percentile(average_bscan,95)] = 1.0


test = np.load(flist[0])
test_amp = np.abs(test)


def smooth(im,width=9):
    kernel = np.zeros((width,width))
    kernel[0,:] = 1.0
    smoothed = np.real(np.fft.ifft2(np.fft.fft2(im,s=im.shape)*np.fft.fft2(kernel,s=im.shape)))
    return smoothed/np.sum(kernel)

#smoothed = smooth(smooth(test_amp))
window_width = 5
smoothed = smooth(test_amp,11)
prof = np.mean(smoothed[:,test.shape[1]//2-window_width//2:test.shape[1]//2+window_width//2],axis=1)
left = prof[:-2]
center = prof[1:-1]
right = prof[2:]
rising = center>left
falling = center>right
peaks = np.where(rising*falling)[0]+1
for peak in peaks:
    isos_idx = peak
    if prof[peak]>7000:
        break

x_start = test.shape[1]//2
current_z = isos_idx

graph = []

rad = 2
z_vec = np.arange(test.shape[0])

for current_x in range(x_start,-1,-1):
    ascan = smoothed[:,current_x]
    atemp = np.zeros(ascan.shape)
    atemp[current_z-rad:current_z+rad+1] = ascan[current_z-rad:current_z+rad+1]
    dz = np.abs(z_vec-current_z)
    score = atemp#/(np.sqrt(dz)+1)
    current_z = np.argmax(score)
    graph.append([current_x,current_z])

current_z = isos_idx
for current_x in range(x_start+1,test.shape[1]):
    ascan = smoothed[:,current_x]
    atemp = np.zeros(ascan.shape)
    atemp[current_z-rad:current_z+rad+1] = ascan[current_z-rad:current_z+rad+1]
    dz = np.abs(z_vec-current_z)
    score = atemp#/(np.sqrt(dz)+1)
    current_z = np.argmax(score)
    graph.append([current_x,current_z])

graph = np.array(graph)
x_vec = graph[:,0]
isos_idx_vec = graph[:,1]

order = np.argsort(x_vec)
x_vec = x_vec[order]
isos_idx_vec = isos_idx_vec[order]



isos_idx_vec_shifted = np.zeros(isos_idx_vec.shape,dtype=np.int16)
for idx,(x,isos_idx) in enumerate(zip(x_vec,isos_idx_vec)):
    segment = test_amp[isos_idx-1:isos_idx+2,x]
    shift = np.argmax(segment)-1
    isos_idx_vec_shifted[idx]=isos_idx_vec[idx]+shift


if True:
    plt.figure()
    plt.imshow(test_amp,cmap='gray')
    plt.autoscale(False)
    plt.plot(x_vec,isos_idx_vec,'y.',alpha=0.2)
    plt.plot(x_vec,isos_idx_vec_shifted,'g.',alpha=0.2)
    plt.show()



sz,sx = test_amp.shape
sy = n_files

fig = plt.figure(figsize=(12,6))
ax1,ax2 = fig.subplots(1,2)

for t,f in zip(t_vec,flist):

    im = np.load(f)
    phase = np.angle(im)+np.pi
    amp = np.abs(im)
    

    ax1.imshow(amp,cmap='gray')
    
    for x in range(sx):
        z = isos_idx_vec_shifted[x]
        ax1.plot(x,z,'y.')
        p0 = phase[z,x]
        phase[:,x] = phase[:,x]-p0
        phase[:,x] = phase[:,x]%(2*np.pi)

    ax2.imshow(mask*phase,cmap='jet')
    plt.title(t)
    plt.pause(.1)
    plt.savefig(os.path.join(output_directory,'phase_%0.3f.png'%t))
    ax1.clear()
    ax2.clear()

sys.exit()

    
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

    if show:
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
        plt.savefig(png_outfn,dpi=100)
    
    npy_outfn = os.path.join(npy_output_directory,'phase_ramp_frames_%02d-%02d.npy'%(block_start,block_start+block_size-1))
    np.save(npy_outfn,average_bscan*np.exp(1j*phase_slope_image))

    if show:
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
