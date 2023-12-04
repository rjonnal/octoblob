import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob
import functions as blobf
from matplotlib.widgets import Button, Slider
import config as cfg
import pathlib
import multiprocessing as mp
import datetime

mp.set_start_method('fork')

#################################### Start of hard coded parameters ###########################
dB_clims = cfg.dB_clims

# parameters shifting histogram method
n_base_bins = cfg.n_base_bins
n_bin_shifts = cfg.n_bin_shifts
histogram_threshold_fraction = cfg.histogram_threshold_fraction
write_pngs = cfg.write_pngs
require_multiprocessing = cfg.require_multiprocessing
block_size = cfg.block_size
bscan_interval = cfg.bscan_interval

#################################### End of hard coded parameters #############################

def xcorr(tup):
    ref = tup[0]
    tar = tup[1]
    nxc = np.abs(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))
    pidx = np.argmax(nxc)
    p = np.max(nxc)
    if pidx>len(tar)//2:
        pidx = pidx-len(tar)
    return p,pidx

def roll_bscan(tup):
    bf = tup[0]
    shift = tup[1]
    print('Flattening %s by %d.'%(bf,shift))
    bscan = np.load(bf)
    bscan = np.roll(bscan,shift,axis=0)
    np.save(bf,bscan)
    return 1        


def save_org(tup,diagnostics=False):
    # this function takes a tuple:
    # tup[0] is a list of filenames corresponding to bm_scans
    # tup[1] is an output index
    # tup[2] is an output root folder
    # the function loads the B-scans, computes ORG stuff, and saves it
    #print(tup)

    block_filenames = tup[0]
    output_index = tup[1]
    root_output_folder = tup[2]
    
    amplitude_output_folder = os.path.join(root_output_folder,'block_amp')
    phase_velocity_output_folder = os.path.join(root_output_folder,'phase_velocity')
    complex_variance_output_folder = os.path.join(root_output_folder,'block_var')
    residual_error_output_folder = os.path.join(root_output_folder,'residual_error')
    os.makedirs(phase_velocity_output_folder,exist_ok=True)
    os.makedirs(amplitude_output_folder,exist_ok=True)
    os.makedirs(complex_variance_output_folder,exist_ok=True)
    os.makedirs(residual_error_output_folder,exist_ok=True)


    
    print('Working on block %s.'%block_filenames)
    block = [np.load(bf) for bf in block_filenames]
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


    amplitude = []
    phase_variance = []
    amplitude_variance = []
    phase_velocity = []
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

        amplitude_variance.append(avar)
        phase_variance.append(pvar)

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

    phase_variance = np.array(phase_variance).T
    amplitude = np.array(amplitude).T
    amplitude_variance = np.array(amplitude_variance).T
    complex_variance = amplitude_variance*np.exp(1j*phase_variance)
    phase_velocity = np.array(phase_velocity).T
    residual_error = np.array(residual_error).T
    
    amplitude_fn = os.path.join(amplitude_output_folder,'%05d.npy'%output_index)
    complex_variance_fn = os.path.join(complex_variance_output_folder,'%05d.npy'%output_index)
    phase_velocity_fn = os.path.join(phase_velocity_output_folder,'%05d.npy'%output_index)
    residual_error_fn = os.path.join(residual_error_output_folder,'%05d.npy'%output_index)
    
    np.save(amplitude_fn,amplitude)
    np.save(phase_velocity_fn,phase_velocity)
    np.save(complex_variance_fn,complex_variance)
    np.save(residual_error_fn,residual_error)
    
    return 1


if __name__=='__main__':
    n_cpus = os.cpu_count()
    
    try:
        filt = sys.argv[1]
    except:
        print('Please supply a file or folder name at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
        sys.exit()

    files = list(pathlib.Path(filt).rglob('*.unp'))
    files = [str(f) for f in files]
    if len(files)==0:
        files = [filt]
        
    for fn in files:
        tag = fn.replace('.unp','')
        bscan_folder = '%s_bscans'%tag

        org_folder = os.path.join(bscan_folder,'org')
        os.makedirs(org_folder,exist_ok=True)
        
        cfg = blobf.get_configuration(fn.replace('.unp','.xml'))
        bscan_files = sorted(glob.glob(os.path.join(bscan_folder,'complex*.npy')))
        if len(bscan_files)==0:
            continue
        test = np.load(bscan_files[0])
        if test.shape[0]==0:
            continue

        N = len(bscan_files)
        first_start = 0
        last_start = N-block_size+1
        

        start_indices = range(first_start,last_start)
        block_filenames = []
        for si in start_indices:
            filenames = bscan_files[si:si+block_size]
            block_filenames.append(filenames)

        output_indices = range(0,len(block_filenames))
        output_folders = [org_folder]*len(output_indices)

        tups = []
        for k in range(0,len(block_filenames)):
            tups.append((block_filenames[k],k,org_folder))

        try:
            p = mp.Pool(n_cpus)
            success = p.map(save_org,tups)
        except Exception as e:
            print(e)
            if require_multiprocessing:
                sys.exit('Multiprocessing failed. Serial processing aborted because require_multiprocessing=True.')
            success = []
            for tup in tups:
                success.append(save_org(tup))


            
