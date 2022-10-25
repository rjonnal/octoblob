import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.io as sio


def centers_to_edges(bin_centers):
    # convert an array of bin centers to bin edges, using the mean
    # spacing of the centers to determine bin width

    # check if sorted:
    assert all(bin_centers[1:]>bin_centers[:-1])

    bin_width = np.mean(np.diff(bin_centers))
    half_width = bin_width/2.0
    first_edge = bin_centers[0]-half_width
    last_edge = bin_centers[-1]+half_width
    return np.linspace(first_edge,last_edge,len(bin_centers)+1)

def bin_shift_histogram(vals,bin_centers,resample_factor=1,diagnostics=False):
    shifts = np.linspace(bin_centers[0]/float(len(bin_centers)),
                          bin_centers[-1]/float(len(bin_centers)),resample_factor)

    #print('shifts:')
    #print(shifts)

    #print('bin centers:')
    #print(bin_centers)
    
    n_shifts = len(shifts)
    n_bins = len(bin_centers)

    all_counts = np.zeros((n_shifts,n_bins))
    all_edges = np.zeros((n_shifts,n_bins+1))

    for idx,s in enumerate(shifts):
        edges = centers_to_edges(bin_centers+s)
        all_counts[idx,:],all_edges[idx,:] = np.histogram(vals,edges)

    all_centers = (all_edges[:,:-1]+all_edges[:,1:])/2.0
    all_counts = all_counts/float(resample_factor)
    all_centers = all_centers

    if diagnostics:
        bin_edges = centers_to_edges(bin_centers)
        bin_width = np.mean(np.diff(bin_edges))
        shift_size = np.mean(np.diff(shifts))
        
        plt.figure(figsize=(3*opf.IPSP,opf.IPSP),dpi=opf.screen_dpi)
        plt.subplot(1,3,1)
        plt.imshow(all_counts)
        plt.title('counts')
        plt.xlabel('bins')
        plt.ylabel('shifts')
        
        #plt.gca().set_yticks(np.arange(0,n_shifts,3))
        #plt.gca().set_yticklabels(['%0.2f'%s for s in shifts])

        #plt.gca().set_xticks(np.arange(0,n_bins,3))
        #plt.gca().set_xticklabels(['%0.2f'%bc for bc in bin_centers])
        plt.colorbar()

        all_counts = all_counts.T
        all_centers = all_centers.T.ravel()

        plt.subplot(1,3,2)
        plt.hist(vals,bin_edges,width=bin_width*0.8)
        plt.title('standard histogram')
        plt.subplot(1,3,3)
        plt.bar(all_centers.ravel(),all_counts.ravel(),width=shift_size*0.8)
        plt.title('bin shifted histogram')

        plt.show()

        
        #save_diagnostics(diagnostics,'bin_shift_histogram')

    return all_counts.T.ravel(),all_centers.T.ravel()

def test_bin_shift_histogram(N=1000,mu=2.5,sigma=30.0):

    s1 = np.random.rand(N)*sigma
    s2 = s1+mu
    noise1 = np.random.randn(N)*2.0
    noise2 = np.random.randn(N)*2.0
    s1 = s1%(np.pi*2)
    s2 = s2%(np.pi*2)

    s1 = s1 + noise1
    s2 = s2 + noise2

    
    vals = s2-s1
    
    resample_factor = 4
    bin_edges_sparse = np.linspace(0,2*np.pi,16)
    bin_edges_dense = np.linspace(0,2*np.pi,16*resample_factor)
    #vals = (vals+np.pi)%(2*np.pi)-np.pi

    counts,centers = bin_shift_histogram(vals,bin_edges_sparse,resample_factor)
    plt.figure()
    plt.subplot(1,3,1)
    plt.hist(vals,bin_edges_sparse)
    plt.title('sparse histogram')
    plt.xlim((0,2*np.pi))
    plt.subplot(1,3,2)
    plt.hist(vals,bin_edges_dense)
    plt.title('dense histogram')
    plt.xlim((0,2*np.pi))
    plt.subplot(1,3,3)
    plt.bar(centers,counts)
    plt.title('resampled histogram')
    plt.xlim((0,2*np.pi))
    plt.show()
    
#test_bin_shift_histogram()

def wrap_into_range(arr,phase_limits=(-np.pi,np.pi)):
    lower,upper = phase_limits
    above_range = np.where(arr>upper)
    below_range = np.where(arr<lower)
    arr[above_range]-=2*np.pi
    arr[below_range]+=2*np.pi
    return arr


def get_phase_jumps(phase_stack,mask,
                    n_bins=16,
                    resample_factor=24,
                    n_smooth=5,polynomial_smoothing=True,diagnostics=False):

    # Take a stack of B-scan phase arrays, with dimensions
    # (z,x,repeats), and return a bulk-motion corrected
    # version

    n_depth = phase_stack.shape[0]
    n_fast = phase_stack.shape[1]
    n_reps = phase_stack.shape[2]
    
    d_phase_d_t = np.diff(phase_stack,axis=2)
    # multiply each frame of the diff array by
    # the mask, so that only valid values remain;
    # Then wrap any values above pi or below -pi into (-pi,pi) interval.
    d_phase_d_t = wrap_into_range(d_phase_d_t)


    if diagnostics:
        plt.figure(figsize=((n_reps-1)*opf.IPSP,2*opf.IPSP),dpi=opf.screen_dpi)
        plt.suptitle('phase shifts between adjacent frames in cluster')
        for rep in range(1,n_reps):
            plt.subplot(2,n_reps-1,rep)
            plt.imshow(d_phase_d_t[:,:,rep-1],aspect='auto')
            if rep==1:
                plt.ylabel('unmasked')
            if rep==n_reps-1:
                plt.colorbar()
            plt.title(r'$d\theta_{%d,%d}$'%(rep,rep-1))
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2,n_reps-1,rep+(n_reps-1))
            plt.imshow(mask*d_phase_d_t[:,:,rep-1],aspect='auto')
            if rep==1:
                plt.ylabel('masked')
            if rep==n_reps-1:
                plt.colorbar()
            plt.xticks([])
            plt.yticks([])
        save_diagnostics(diagnostics,'phase_shifts')
            
    d_phase_d_t = np.transpose(np.transpose(d_phase_d_t,(2,0,1))*mask,(1,2,0))

    
    bin_edges = np.linspace(-np.pi,np.pi,n_bins)
    
    # The key idea here is from Makita, 2006, where it is well explained. In
    # addition to using the phase mode, we also do bin-shifting, in order to
    # smooth the histogram. Again departing from Justin's approach, let's
    # just specify the top level bins and a resampling factor, and let the
    # histogram function do all the work of setting the shifted bin edges.

    b_jumps = np.zeros((d_phase_d_t.shape[1:]))
    bin_counts = np.zeros((d_phase_d_t.shape[1:]))

    if diagnostics:
        plt.figure(figsize=((n_reps-1)*opf.IPSP,1*opf.IPSP),dpi=opf.screen_dpi)
        total_bins = n_bins*resample_factor
        hist_sets = np.zeros((n_reps-1,n_fast,total_bins))

    for f in range(n_fast):
        valid_idx = np.where(mask[:,f])[0]
        for r in range(n_reps-1):
            vals = d_phase_d_t[valid_idx,f,r]
            if diagnostics:
                # RSJ, 23 March 2020:
                # We'll add a simple diagnostic here--a printout of the number of samples and the
                # interquartile range. These can be used to determine the optimal bin width, following
                # Makita et al. 2006 "Optical coherence angiography" eq. 3.
                try:
                    q75,q25 = np.percentile(vals,(75,25))
                    IQ = q75-q25
                    m = float(len(vals))
                    h = 2*IQ*m**(-1/3)
                    n_bins = np.ceil(2*np.pi/h)
                    bscan_indices = '%01d-%01d'%(r,r+1)
                    log_diagnostics(diagnostics,'histogram_optimal_bin_width',
                                    header=['bscan indices','ascan index','IQ','m','h','n_bins'],
                                    data=[bscan_indices,f,IQ,m,h,n_bins],
                                    fmt=['%s','%d','%0.3f','%d','%0.3f','%d'],clobber=f==0)
                except IndexError:
                    pass
                
            # if it's the first rep of the first frame, and diagnostics are requested, print the histogram diagnostics
            if f==0 and r==0:
                [counts,bin_centers] = bin_shift_histogram(vals,bin_edges,resample_factor,diagnostics=diagnostics)
            else:
                [counts,bin_centers] = bin_shift_histogram(vals,bin_edges,resample_factor,diagnostics=False)
                
            if diagnostics:
                hist_sets[r,f,:] = counts
            bulk_shift = bin_centers[np.argmax(counts)]
            bin_count = np.max(counts)
            b_jumps[f,r] = bulk_shift
            bin_counts[f,r] = bin_count

    #if polynomial_smoothing:
    #    polynomial_smooth_phase(bin_counts,b_jumps)
            
    if diagnostics:
        for idx,hist_set in enumerate(hist_sets):
            plt.subplot(1,n_reps-1,idx+1)
            plt.imshow(hist_set,interpolation='none',aspect='auto',extent=(np.min(bin_centers),np.max(bin_centers),0,n_fast-1),cmap='gray')
            plt.yticks([])
            plt.xlabel(r'$d\theta_{%d,%d}$'%(idx+1,idx))
            if idx==0:
                plt.ylabel('fast scan index')
            plt.colorbar()
            plt.autoscale(False)
            plt.plot(b_jumps[:,idx],range(n_fast)[::-1],'g.',alpha=0.2)
        plt.suptitle('shifted bulk motion histograms (count)')
        save_diagnostics(diagnostics,'bulk_motion_histograms')

        # now let's show some examples of big differences between adjacent A-scans
        legendfontsize = 8
        lblfmt = 'r%d,x%d'
        pts_per_example = 10
        max_examples = 16
        dtheta_threshold = np.pi/2.0
        fig = plt.figure(figsize=(opf.IPSP*4,opf.IPSP*4))
        example_count = 0
        for rep_idx,hist_set in enumerate(hist_sets):
            temp = np.diff(b_jumps[:,rep_idx])
            worrisome_indices = np.where(np.abs(temp)>dtheta_threshold)[0]
            # index n in the diff means the phase correction difference between
            # scans n+1 and n
            for bad_idx,scan_idx in enumerate(worrisome_indices):
                example_count = example_count + 1
                if example_count>max_examples:
                    break
                plt.subplot(4,4,example_count)
                mask_line = mask[:,scan_idx]
                voffset = False
                vals = np.where(mask_line)[0][:pts_per_example]
                plt.plot(phase_stack[vals,scan_idx,rep_idx],'rs',label=lblfmt%(rep_idx,scan_idx))
                plt.xticks([])
                plt.plot(phase_stack[vals,scan_idx,rep_idx+1]+voffset*2*np.pi,'gs',label=lblfmt%(rep_idx+1,scan_idx))
                plt.xticks([])
                plt.plot(phase_stack[vals,scan_idx+1,rep_idx]+voffset*4*np.pi,'bs',label=lblfmt%(rep_idx,scan_idx+1))
                plt.xticks([])
                plt.plot(phase_stack[vals,scan_idx+1,rep_idx+1]+voffset*6*np.pi,'ks',label=lblfmt%(rep_idx+1,scan_idx+1))
                plt.xticks([])
                plt.legend(bbox_to_anchor=(0,-0.2,1,0.2), loc="upper left",
                           mode="expand", borderaxespad=0, ncol=4, fontsize=legendfontsize)
                plt.title(r'$d\theta=%0.1f$ rad'%temp[scan_idx])
            if example_count>=max_examples:
                break
        plt.suptitle('scans involved in each mode jump')
        save_diagnostics(diagnostics,'histogram_mode_examples_bad')


        # now let's show some examples of small differences between adjacent A-scans
        dtheta_threshold = np.pi/20.0
        fig = plt.figure(figsize=(opf.IPSP*4,opf.IPSP*4))
        example_count = 0
        for rep_idx,hist_set in enumerate(hist_sets):
            temp = np.diff(b_jumps[:,rep_idx])
            good_indices = np.where(np.abs(temp)<dtheta_threshold)[0]
            # index n in the diff means the phase correction difference between
            # scans n+1 and n
            for bad_idx,scan_idx in enumerate(good_indices):
                example_count = example_count + 1
                if example_count>max_examples:
                    break
                plt.subplot(4,4,example_count)
                mask_line = mask[:,scan_idx]
                voffset = False
                vals = np.where(mask_line)[0][:pts_per_example]
                plt.plot(phase_stack[vals,scan_idx,rep_idx],'rs',label=lblfmt%(rep_idx,scan_idx))
                plt.xticks([])
                plt.plot(phase_stack[vals,scan_idx,rep_idx+1]+voffset*2*np.pi,'gs',label=lblfmt%(rep_idx+1,scan_idx))
                plt.xticks([])
                plt.plot(phase_stack[vals,scan_idx+1,rep_idx]+voffset*4*np.pi,'bs',label=lblfmt%(rep_idx,scan_idx+1))
                plt.xticks([])
                plt.plot(phase_stack[vals,scan_idx+1,rep_idx+1]+voffset*6*np.pi,'ks',label=lblfmt%(rep_idx+1,scan_idx+1))
                plt.xticks([])
                plt.legend(bbox_to_anchor=(0,-0.2,1,0.2), loc="upper left",
                           mode="expand", borderaxespad=0, ncol=4, fontsize=legendfontsize)
                plt.title(r'$d\theta=%0.1f$ rad'%temp[scan_idx])
            if example_count>=max_examples:
                break
        plt.suptitle('scans involved in each mode jump')
        save_diagnostics(diagnostics,'histogram_mode_examples_good')
        
        
    # Now unwrap to prevent discontinuities (although this may not impact complex variance)
    b_jumps = np.unwrap(b_jumps,axis=0)

    # Smooth by convolution. Don't forget to divide by kernel size!
    # b_jumps = sps.convolve2d(b_jumps,np.ones((n_smooth,1)),mode='same')/float(n_smooth)

    return b_jumps

if __name__=='__main__':
    test_bin_shift_histogram()
