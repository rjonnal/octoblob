import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob
import functions as blobf
from matplotlib.widgets import Button, Slider
import config as cfg

try:
    bscan_folder = sys.argv[1]
except:
    print('Please supply the bscan folder at the command line, i.e., python mwe_step_03_make_org_blocks.py XX_YY_ZZ_bscans')
    sys.exit()

#################################### Start of hard coded parameters ###########################
dB_clims = cfg.dB_clims

phase_velocity_png_contrast_percentiles = cfg.phase_velocity_png_contrast_percentiles
amplitude_png_contrast_percentiles = cfg.amplitude_png_contrast_percentiles
variance_png_contrast_percentiles = cfg.variance_png_contrast_percentiles
residual_error_png_contrast_percentiles = cfg.residual_error_png_contrast_percentiles

block_size = cfg.block_size
bscan_interval = cfg.bscan_interval
reference_bscan_filename = cfg.reference_bscan_filename

# parameters shifting histogram method
n_base_bins = cfg.n_base_bins
n_bin_shifts = cfg.n_bin_shifts
histogram_threshold_fraction = cfg.histogram_threshold_fraction
write_pngs = cfg.write_pngs

#################################### End of hard coded parameters #############################

tag = bscan_folder.replace('_bscans/','')

diagnostics = blobf.Diagnostics(tag)
# diagnostics = False # if you want to skip the diagnostics

org_folder = os.path.join(bscan_folder,'org')
os.makedirs(org_folder,exist_ok=True)

# In the ORG folder we need three subfolders for phase slope, block average amplitude, and real and imaginary variances,
# which are stored as a single complex B-scan (phasor notation).
phase_velocity_folder = os.path.join(org_folder,'phase_velocity')
block_amp_folder = os.path.join(org_folder,'block_amp')
block_var_folder = os.path.join(org_folder,'block_var')
residual_error_folder = os.path.join(org_folder,'residual_error')

os.makedirs(phase_velocity_folder,exist_ok=True)
os.makedirs(block_amp_folder,exist_ok=True)
os.makedirs(block_var_folder,exist_ok=True)
os.makedirs(residual_error_folder,exist_ok=True)

bscan_files = glob.glob(os.path.join(bscan_folder,'complex*.npy'))
bscan_files.sort()

reference_bscan_index = bscan_files.index(os.path.join(bscan_folder,reference_bscan_filename))

bscans = []
for f in bscan_files:
    bscans.append(np.load(f))

N = len(bscan_files)

# time taken by each block, for calculating phase velocity
dt = (block_size-1)*bscan_interval

def get_z_crop_coords(bscan,inner_border=20,outer_border=0,noise_level=0.05,diagnostics=False):
    prof = np.mean(np.abs(bscan),axis=1)
    thresh = np.max(prof)*noise_level
    valid = np.where(prof>thresh)[0]
    z2,z1 = valid[-1]+outer_border,valid[0]-inner_border
    z2 = min(bscan.shape[0],z2)
    z1 = max(z1,0)
    if diagnostics:
        fig = diagnostics.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(prof)
        ax1.axhline(thresh)
        ax1.axvline(z1)
        ax1.axvline(z2)
        diagnostics.save()
    return z2,z1

def dB(bscan):
    return 20*np.log10(np.abs(bscan))

# average all B-scans and get automatic cropping coordinates
#bscan_mean = np.mean(np.abs(np.array(bscans)),axis=0)
#crop_z2,crop_z1 = get_z_crop_coords(bscan_mean,diagnostics=diagnostics)

# crop the B-scans to make them easier to work with
#bscans = [b[crop_z1:crop_z2,:] for b in bscans]
#bscan_mean_cropped = np.mean(np.abs(np.array(bscans)),axis=0)

reference_bscan = bscans[reference_bscan_index]
reference_profile = np.mean(np.abs(reference_bscan),axis=1)

# if diagnostics:
#     label = 'bscan_auto_crop'
#     fig = diagnostics.figure(label=label,figsize=(8,6))
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
    
#     ax1.imshow(dB(bscan_mean),cmap='gray')
#     ax1.set_title('mean bscan before cropping')
#     ax2.imshow(dB(bscan_mean_cropped),cmap='gray')
#     ax2.set_title('mean bscan after cropping')
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     diagnostics.save()

def register_profiles(tar,ref):
    nxc = np.abs(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))
    pidx = np.argmax(nxc)
    
    if pidx>len(tar)//2:
        pidx = pidx-len(tar)
    return pidx

def get_profile(bscan):
    return np.mean(np.abs(bscan),axis=1)

def roll_bscan(reference_profile,bscan):
    shift = register_profiles(reference_profile,get_profile(bscan))
    bscan = np.roll(bscan,shift,axis=0)
    return bscan

def align_bscans_axially(bscan_list,diagnostics=False):
    unaligned_mean = np.mean(np.abs(np.array(bscan_list)),axis=0)
    bscan_list = [roll_bscan(reference_profile,b) for b in bscan_list]
    aligned_mean = np.mean(np.abs(np.array(bscan_list)),axis=0)
    
    if diagnostics:
        fig = diagnostics.figure(figsize=(8,8))
        ax = fig.subplots(2,2)
        ax[0,0].imshow(dB(unaligned_mean),cmap='gray')
        ax[0,0].set_title('unaligned')
        ax[0,1].plot(np.mean(unaligned_mean,axis=1))
        
        ax[1,0].imshow(dB(aligned_mean),cmap='gray')
        ax[1,0].set_title('aligned')
        ax[1,1].plot(np.mean(aligned_mean,axis=1))
        diagnostics.save()
    return bscan_list

bscans = align_bscans_axially(bscans,diagnostics=diagnostics)


first_start = 0
last_start = N-block_size
# working with frames 80 - 140; stimulus at frame 100
diagnostic_histogram_count = 0

for start_idx in range(first_start,last_start):
    print('Working on block %d of %d.'%(start_idx,last_start))
    output_index = start_idx
    
    end_idx = start_idx+block_size
    block = bscans[start_idx:end_idx]

    block = np.array(block)
    
    # block shape is block_size, n_depth, n_fast

    # 1. Average the block in time to get an average amplitude B-scan
    temp = np.nanmean(np.abs(block),axis=0)
    
    # 2. The next step is bulk-motion correction
    # 2a. Only bright pixels are used for bulk-motion correction. We'll use all pixels that are at least 10%
    #     of the image max.
    mask = np.zeros(temp.shape)
    
    mask[temp>np.max(temp)*histogram_threshold_fraction] = 1
    if diagnostics:
        fig = diagnostics.figure(label='bulk_motion_correction_mask')
        ax = fig.add_subplot(111)
        ax.imshow(mask)
        diagnostics.save(fig)

    # 2b. Now we work our way across the B-scan and do bulk motion correction for each set of 5 sister A-scans;
    #     we use the same mask for each B-scan, obviously. And we measure bulk motion relative to the first
    #     B-scan in the block.
    n_bscans,n_depth,n_fast = block.shape
    
    abs_all_counts = {}
    for k in range(n_bscans):
        abs_all_counts[k] = []
        
    rel_all_counts = {}
    for k in range(n_bscans):
        rel_all_counts[k] = []


    # We need to create the following three outputs: phase velocity, amplitude, complex variance, and residual error;
    # each one is assembled from computations (polyfit, absolute mean, var, and rms) performed on sister
    # ascans. As we iterate through the sets of sisters, we'll build these as lists and convert to arrays at the end.


    phase_velocity = []
    amplitude = []
    complex_variance = []
    residual_error = []
    
    
    for f in range(n_fast):
        
        mask_column = mask[:,f]
        abs_reference_ascan = block[0,:,f]
        abs_reference_pixels = abs_reference_ascan[np.where(mask_column)]
        rel_phase_shifts = np.zeros(len(abs_reference_pixels))

        cumulative_rel_ascan_phase_shift = 0.0
        
        for step in range(1,n_bscans):
            rel_reference_ascan = block[step-1,:,f]
            rel_reference_pixels = rel_reference_ascan[np.where(mask_column)]

            target_ascan = block[step,:,f]
            target_pixels = target_ascan[np.where(mask_column)]

            abs_phase_shifts = np.angle(target_pixels)-np.angle(abs_reference_pixels)
            abs_phase_shifts = np.unwrap(abs_phase_shifts)%(np.pi*2)


            # IMPORTANT: the rel_phase_shifts will not have to be integrated (cumsum) because we are taking the difference
            # between the target pixels and the rel_reference_pixels, which have already been corrected at the end of this
            # loop.
            
            rel_phase_shifts = np.angle(target_pixels)-np.angle(rel_reference_pixels)
            rel_phase_shifts = np.unwrap(rel_phase_shifts)%(np.pi*2)
            
            
            # Now we have the phase shifts between the target and reference pixels; these should tell us how much
            # the target A-scan has moved relative to the reference A-scan. Typically, this phase shift is corrected
            # by using the resampling histogram method proposed by Makita 2006 "Optical coherence angiography".

            full_range = 2*np.pi

            base_bin_width = full_range/n_base_bins
            base_bin_centers = np.arange(0.0,full_range,base_bin_width)
            
            base_bin_starts = base_bin_centers-base_bin_width/2.0
            base_bin_ends = base_bin_centers+base_bin_width/2.0

            base_bin_edges = np.array(list(base_bin_starts) + [base_bin_ends[-1]])
            
            shift_size = base_bin_width/n_bin_shifts
            
            abs_resampled_centers = []
            abs_resampled_counts = []
            rel_resampled_centers = []
            rel_resampled_counts = []

            if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                fig = diagnostics.figure(figsize=(6,n_bin_shifts//3),label='shifting_histogram')
                ax = fig.subplots(n_bin_shifts+1,1)
                
            for n_shift in range(n_bin_shifts):
                # to use numpy hist we must specify bin edges including the rightmost edge
                bin_edges = base_bin_edges + n_shift*shift_size
                abs_counts,abs_edges = np.histogram(abs_phase_shifts,bins=bin_edges)
                abs_centers = (abs_edges[1:]+abs_edges[:-1])/2.0
                
                abs_resampled_centers = abs_resampled_centers+list(abs_centers)
                abs_resampled_counts = abs_resampled_counts+list(abs_counts)

                rel_counts,rel_edges = np.histogram(rel_phase_shifts,bins=bin_edges)
                rel_centers = (rel_edges[1:]+rel_edges[:-1])/2.0
                rel_resampled_centers = rel_resampled_centers+list(rel_centers)
                rel_resampled_counts = rel_resampled_counts+list(rel_counts)
                if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                    ax[n_shift].bar(abs_centers,abs_counts,width=base_bin_width,linewidth=1,edgecolor='k',alpha=0.5,label='abs')
                    ax[n_shift].bar(rel_centers,rel_counts,width=base_bin_width,linewidth=1,edgecolor='k',alpha=0.5,label='rel')
                    ax[n_shift].set_xlim((0,2*np.pi))
                

            if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                plt.suptitle('bin_shifted_histograms')
                #plt.legend()
                diagnostics.save(ignore_limit=True)

            abs_order = np.argsort(abs_resampled_centers)
            abs_resampled_counts = np.array(abs_resampled_counts)[abs_order]
            abs_resampled_centers = np.array(abs_resampled_centers)[abs_order]
            abs_all_counts[step-1].append(abs_resampled_counts)
            
            rel_order = np.argsort(rel_resampled_centers)
            rel_resampled_counts = np.array(rel_resampled_counts)[rel_order]
            rel_resampled_centers = np.array(rel_resampled_centers)[rel_order]
            rel_all_counts[step-1].append(rel_resampled_counts)

            if diagnostics and start_idx==first_start and diagnostic_histogram_count<8:
                fig = diagnostics.figure(label='resampled_histogram')
                ax = fig.add_subplot(111)
                ax.bar(abs_resampled_centers,abs_resampled_counts,width=base_bin_width/n_bin_shifts,linewidth=0.25,edgecolor='k',alpha=0.5,label='abs')
                ax.bar(rel_resampled_centers,rel_resampled_counts,width=base_bin_width/n_bin_shifts,linewidth=0.25,edgecolor='k',alpha=0.5,label='rel')
                ax.set_xlim((0,2*np.pi))
                ax.legend()
                ax.set_title('resampled sister phase differences\nblock %d, fast %d, sister %d'%(start_idx,f,step))
                diagnostics.save(ignore_limit=True)
                diagnostic_histogram_count += 1
                

            abs_winners = abs_resampled_centers[np.where(abs_resampled_counts==np.max(abs_resampled_counts))]
            abs_winners = np.unwrap(abs_winners)
            abs_ascan_phase_shift = np.median(abs_winners)

            rel_winners = rel_resampled_centers[np.where(rel_resampled_counts==np.max(rel_resampled_counts))]
            rel_winners = np.unwrap(rel_winners)
            rel_ascan_phase_shift = np.median(rel_winners)

            #print(abs_ascan_phase_shift,cumulative_rel_ascan_phase_shift)
            
            if False:
                plt.figure()
                plt.subplot(2,1,1)
                plt.bar(abs_resampled_centers,abs_resampled_counts,width=base_bin_width/n_bin_shifts,linewidth=0.25,edgecolor='k')
                plt.xlim((0,2*np.pi))
                plt.title('absolute shift histogram')
                plt.subplot(2,1,2)
                plt.bar(rel_resampled_centers,rel_resampled_counts,width=base_bin_width/n_bin_shifts,linewidth=0.25,edgecolor='k')
                plt.xlim((0,2*np.pi))
                plt.title('relative shift histogram')
                plt.show()
            
            #ascan_phase_shift = abs_resampled_centers[np.argmax(abs_resampled_counts)]

            #print(abs_ascan_phase_shift,rel_ascan_phase_shift)
            
            block[step,:,f] = block[step,:,f] * np.exp(-1j*rel_ascan_phase_shift)
            if False:
                test_pixels = block[step,:,f][np.where(mask_column)]
                print('pre-correction correlation:',np.corrcoef(np.angle(abs_reference_pixels),np.angle(target_pixels))[1,0])
                print('post-correction correlation:',np.corrcoef(np.angle(abs_reference_pixels),np.angle(test_pixels))[1,0])


        sisters = block[:,:,f]
        
        # now we have the bulk-motion corrected sisters, let's add to our growing B-scans
        sisters_mean_amplitude = np.nanmean(np.abs(sisters),axis=0)
        amplitude.append(sisters_mean_amplitude)
        
        sisters_amplitude = np.abs(sisters)
        sisters_phase = np.angle(sisters)
        sisters_phase = np.unwrap(sisters_phase,axis=0)
        
        # variance is being stored in a kind of funny way, to permit downstream computation
        # of phase variance and amplitude variance, as needed:
        avar = np.var(sisters_amplitude,axis=0)
        pvar = np.var(sisters_phase,axis=0)
        cvar = avar*np.exp(1j*pvar)
        cvar = np.squeeze(cvar)
        complex_variance.append(cvar)
        
        # compute linear fits of sister phase at each depth:
        x = np.arange(n_bscans)*bscan_interval
        p = np.polyfit(x,sisters_phase,1)
        phase_velocity.append(p[0])

        residual_error_ascan = []
        for z in range(sisters.shape[1]):
            fit = np.polyval(p[:,z],x)
            err = np.sqrt(np.sum((sisters_phase[:,z]-fit)**2))
            residual_error_ascan.append(err)

        residual_error.append(np.array(residual_error_ascan))

    phase_velocity = np.array(phase_velocity).T
    amplitude = np.array(amplitude).T
    complex_variance = np.array(complex_variance).T
    residual_error = np.array(residual_error).T

    np.save(os.path.join(phase_velocity_folder,'%05d.npy'%output_index),phase_velocity)
    np.save(os.path.join(block_amp_folder,'%05d.npy'%output_index),amplitude)
    np.save(os.path.join(block_var_folder,'%05d.npy'%output_index),complex_variance)
    np.save(os.path.join(residual_error_folder,'%05d.npy'%output_index),residual_error)
    
    if write_pngs:
        if start_idx==first_start:
            pv_clims = np.percentile(phase_velocity,phase_velocity_png_contrast_percentiles)
            ba_clims = np.percentile(amplitude,amplitude_png_contrast_percentiles)
            bv_clims = np.percentile(np.abs(complex_variance),variance_png_contrast_percentiles)
            re_clims = np.percentile(residual_error,residual_error_png_contrast_percentiles)
            
        plt.figure()
        plt.imshow(phase_velocity,clim=pv_clims)
        plt.savefig(os.path.join(phase_velocity_folder,'%05d.png'%output_index))
        plt.clf()
        plt.imshow(amplitude,clim=ba_clims)
        plt.savefig(os.path.join(block_amp_folder,'amp_%05d.png'%output_index))
        plt.clf()
        plt.imshow(dB(amplitude),clim=dB_clims)
        plt.savefig(os.path.join(block_amp_folder,'dB_%05d.png'%output_index))
        plt.clf()
        plt.imshow(np.abs(complex_variance),clim=bv_clims)
        plt.savefig(os.path.join(block_var_folder,'%05d.png'%output_index))
        plt.clf()
        plt.imshow(residual_error,clim=re_clims)
        plt.savefig(os.path.join(residual_error_folder,'%05d.png'%output_index))
        plt.close()

