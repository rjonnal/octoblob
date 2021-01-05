from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.io as sio
from octoblob import bmp_tools

class ProcessingParameters:

    def __init__(self):
        # This class contains parameters for the processing
        # pipeline. It is meant to be frozen in releases of
        # octoblob, such that processing can be reproduced
        # perfectly without having to fiddle with them.

        # coefficients for resampling lambda into k
        # these coefficients c specify a polynomial p:
        # p(x) = c_0*x^3 + c_1*x^2 + c_2*x + c_3
        # p(x) is a the sampling error in x,
        # and the measured spectra are interpolated from
        # x -> x+p(x)
        self.k_resampling_coefficients = [12.5e-10,-12.5e-7,0,0]

        # these are the coefficients for the unit-amplitude
        # phasor used for removing dispersion chirp; if the
        # coefficients are c, then
        # p(x) = c_0*x^3 + c_1*x^2 + c_2*x + c_3
        # the dechirping phasor D is given by:
        # D = e^[-i*p(x)]
        # the spectra are dechirped by:
        # dechirped_spectrum = spectra*D
        self.dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]

        # the width of the window for gaussian windowing:
        self.gaussian_window_sigma = 0.9

        # paramters for bulk motion estimation, including
        # smoothing by shifting bin edges; see Makita, 2006
        # for a detailed description of the approach;
        # in short, we do a histogram of the B-scan to B-scan
        # phase error, with a fixed number of bins (n_bins);
        # then, we shift the bin edges by a fraction of a
        # bin width and recompute the histogram; the fractional
        # shift is equal to 1/resample_factor
        self.bulk_motion_n_bins = 16
        self.bulk_motion_resample_factor = 24
        self.bulk_motion_n_smooth = 5

        # critical parameters: thresholds for bulk motion correction
        # and phase variance computation
        self.bulk_correction_threshold = 0.3
        self.phase_variance_threshold = 0.43

pp = ProcessingParameters()

class OCTRawData:

    def __init__(self,filename,n_vol,n_slow,n_fast,n_depth,
                 n_repeats=1,dtype=np.uint16,dc_crop=50,
                 fbg_position=None,spectrum_start=None,spectrum_end=None,
                 bit_shift_right=0,n_skip=0,fbg_sign=1):
        
        self.dtype = dtype
        self.n_vol = n_vol
        self.n_slow = n_slow
        self.n_fast = n_fast
        self.n_depth = n_depth
        self.n_repeats = n_repeats
        self.bytes_per_pixel = self.dtype(1).itemsize
        self.n_bytes = self.n_vol*self.n_slow*self.n_fast*self.n_depth*self.bytes_per_pixel
        self.filename = filename
        self.has_fbg = not fbg_position is None
        self.fbg_position = fbg_position
        self.bit_shift_right = bit_shift_right
        self.n_skip = n_skip
        self.fbg_sign = fbg_sign
        
        if spectrum_start is None:
            self.spectrum_start = 0
        else:
            self.spectrum_start = spectrum_start
            
        if spectrum_end is None:
            self.spectrum_end = 0
        else:
            self.spectrum_end = spectrum_end
        
        file_size = os.stat(self.filename).st_size
        skip_bytes = self.n_skip*self.n_depth*self.bytes_per_pixel
        
        try:
            assert file_size==self.n_bytes
            print('Data source established:')
            self.print_volume_info()
            print()
            
        except AssertionError as ae:
            print('File size incorrect.\n%d\texpected\n%d\tactual'%(self.n_bytes,file_size))
            self.print_volume_info()

    def print_volume_info(self):
        print('n_vol\t\t%d\nn_slow\t\t%d\nn_repeats\t%d\nn_fast\t\t%d\nn_depth\t\t%d\nbytes_per_pixel\t%d\ntotal_expected_size\t%d'%(self.n_vol,self.n_slow,self.n_repeats,self.n_fast,self.n_depth,self.bytes_per_pixel,self.n_bytes))


    def align_to_fbg(self,frame,region_height=48,smoothing_size=5,sign=1,do_plots=False):
        # The algorithm here is copied from Justin Migacz's MATLAB prototype; one
        # key difference is that Justin cats 5 sets of spectra together, such that
        # they share the same k-axis; this step is performed on a "compound" frame,
        # with shape (n_k,n_fast*n_repeats)
        if not self.has_fbg:
            return frame
        z1 = self.fbg_position-region_height//2
        z2 = self.fbg_position+region_height//2-1

        if do_plots:
            plt.figure()
            plt.imshow(frame,cmap='gray',aspect='auto')
            plt.axhspan(z1,z2,alpha=0.2)
            plt.show()
        
        
        # crop the relevant region:
        fbg_region = np.zeros((z2-z1,frame.shape[1]))
        fbg_region[:,:] = frame[z1:z2,:]

        # smooth with a kernel of 1x2 pixels
        # use valid region (MATLAB default) to avoid huge gradients
        # at edges
        fbg_region = sps.convolve2d(fbg_region,np.ones((smoothing_size,1)),mode='valid')
        
        # do the vertical derivative:
        fbg_region_derivative = sign*np.diff(fbg_region,axis=0)

        # find the index of maximum rise for each column in the derivative:
        # the +1 at the end is just to square this with Justin's code;
        # ideally, get rid of the +1 here and add back the zero-centering below
        max_rise_index = np.argsort(fbg_region_derivative,axis=0)[-1,:].astype(np.int)+1

        # zero-center the max_rise_index to avoid rolling more than we need to;
        # this departs a bit from Justin's approach, but not consequentially
        # consider adding this back in after the code has been thoroughly checked
        # max_rise_index = max_rise_index - int(np.round(np.mean(max_rise_index)))

        # roll the columns to align
        # Notes:
        # 1. After testing, replace this with an in-place operation,
        #    e.g. frame[:,x] = np.roll(....
        # 2. Ideally, get rid of the loop--it's not necessary; use
        #    advanced indexing, but verify that it's CPU-vectorized,
        #    o.w. leave as is here for clarity

        out = np.zeros(frame.shape)
        for x in range(frame.shape[1]):
            out[:,x] = np.roll(frame[:,x],-max_rise_index[x])

        return out
        
    def get_frame(self,frame_index,volume_index=0,plot_fbg=False):
        '''Get a raw frame from a UNP file. This function will
        try to read configuration details from a UNP file with
        the same name but .xml extension instead of .unp.
        Parameters:
            frame_index: the index of the desired frame; must
              include skipped volumes if file contains multiple
              volumes, unless volume_index is provided
        Returns:
            a 2D numpy array of size n_depth x n_fast
        '''
        frame = None
        # open the file and read in the b-scan
        with open(self.filename,'rb') as fid:
            # Identify the position (in bytes) corresponding to the start of the
            # desired frame; maintain volume_index for compatibility with functional
            # OCT experiments, which have multiple volumes.
            position = volume_index * self.n_depth * self.n_fast * self.n_slow * self.bytes_per_pixel + frame_index * self.n_depth * self.n_fast * self.bytes_per_pixel + self.n_skip * self.n_depth * self.bytes_per_pixel

            
            # Skip to the desired position for reading.
            fid.seek(position,0)

            # Use numpy fromfile to read raw data.
            frame = np.fromfile(fid,dtype=self.dtype,count=self.n_depth*self.n_fast)
            
            # Bit-shift if necessary, e.g. for Axsun data
            if self.bit_shift_right:
                frame = np.right_shift(frame,self.bit_shift_right)

            # Reshape into the k*x 2D array
            frame = frame.reshape(self.n_fast,self.n_depth).T

            # If there's an fbg, align spectra using the align_to_fbg function
            if self.has_fbg:
                frame = self.align_to_fbg(frame,sign=self.fbg_sign,do_plots=plot_fbg)

            frame = frame[self.spectrum_start:self.spectrum_end,:]
        return frame

def dc_subtract(spectra):
    """Estimate DC by averaging spectra spatially (dimension 1),
    then subtract by broadcasting."""
    dc = spectra.mean(1)
    out = (spectra.T-dc).T
    return out

def k_resample(spectra,coefficients=pp.k_resampling_coefficients):
    """Resample the spectrum such that it is uniform w/r/t k.
    The default coefficients here were experimentally determined
    by Justin Migacz, using the Axsun light source.
    Notes:
      1. The coefficients here are for a polynomial defined on
         pixels, so they're physically meaningless. It would be
         better to define our polynomials on k, because then
         we could more easily quantify and compare the chirps
         of multiple light sources, for instance. Ditto for the
         dispersion compensation code.
      2. Justin chose spline interpolation, but I doubt if it's
         different from linear, and may be slower. 'spline' in
         MATLAB means 'cubic spline'.
    """
    # x_in specified on 1..N+1 to accord w/ Justin's code
    # fix this later, ideally as part of a greater effort
    # to define our meshes for mapping and dispersion compensation
    # on k instead of integer index
    x_in = np.arange(1,spectra.shape[0]+1)
    error = np.polyval(coefficients,x_in)
    x_out = x_in + error
    interpolator = spi.interp1d(x_in,spectra,axis=0,kind='cubic',fill_value='extrapolate')
    interpolated = interpolator(x_out)
    # the next step is just to square the output of this with Justin's
    # definitely remove this later
    interpolated[0,:] = interpolated[1,:]
    interpolated[-1,:] = interpolated[-2,:]
    return interpolated

def dispersion_compensate(spectra,coefficients=pp.dispersion_coefficients,diagnostics=False):
    # x_in specified on 1..N+1 to accord w/ Justin's code
    # fix this later, ideally as part of a greater effort
    # to define our meshes for mapping and dispersion compensation
    # on k instead of integer index
    x = np.arange(1,spectra.shape[0]+1)
    dechirping_phasor = np.exp(-1j*np.polyval(coefficients,x))
    dechirped = (spectra.T*dechirping_phasor).T
    if diagnostics:
        before = 20*np.log10(np.abs(np.fft.fft(spectra,axis=0)))
        after = 20*np.log10(np.abs(np.fft.fft(dechirped,axis=0)))
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(before,cmap='gray',aspect='auto',clim=[40,80])
        plt.colorbar()
        plt.title('before disp. comp. (dB)')
        plt.subplot(1,2,2)
        plt.imshow(after,cmap='gray',aspect='auto',clim=[40,80])
        plt.colorbar()
        plt.title('after disp. comp. (dB)')
    return dechirped
    

def gaussian_window(spectra,sigma=pp.gaussian_window_sigma):
    # WindowMat = repmat(exp(-((linspace(-1,1,size(Aspectra,1)))'.^2)/SIG),[1,C*D2]);
    x = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma))
    return (spectra.T*x).T

def spectra_to_bscan(spectra,oversampled_size=None,z1=None,z2=None,x1=None,x2=None,diagnostics=False):
    # pass oversampled_size to fft.fft as parameter n, which will
    # produce the desired behavior in the absense of oversampling,
    # viz. return the original size
    #
    # z1 and z2 correspond to Justin's FRONTCROP and ENDCROP values,
    # but here we expect the values to be pre-scaled by the oversampling
    # factor.
    # 
    # We can leave these as Nones if that's what's passed in, because
    # None as a slice index defaults to the original start/end indices
    #return np.fft.fft(spectra,axis=0,n=oversampled_size)[z1:z2]
    bscan = np.fft.fft(spectra,axis=0,n=oversampled_size)
    if diagnostics:
        plt.figure()
        plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray',clim=[40,80],aspect='auto')
        plt.colorbar()
        plt.axhline(z1)
        if z2>=0:
            plt.axhline(z2)
        else:
            plt.axhline(bscan.shape[0]+z2)
        plt.title('diagnostics: cropped region, contrast limited to (40,80) dB')
    return bscan[z1:z2]

    
def reshape_repeats(multi_bscan,n_repeats,x1=None,x2=None):
    sy,sx = multi_bscan.shape
    # fail if the multi_bscan isn't a multiple of n_repeats
    assert (sx/float(n_repeats))%1==0
    new_sx = sx//n_repeats
    return np.transpose(np.reshape(multi_bscan,(sy,n_repeats,new_sx)),(0,2,1))[:,x1:x2,:]

def show_bscan(bscan,title='',clim=None,plot_ascan=False):
    display_bscan = 20*np.log10(np.abs(bscan))
    # automatic contrast scaling--not ideal because different tissues/lesions require
    # slightly different contrast for optimal visualization, but this will work in a pinch:
    # scale the log image between (mean - 0.5*std, mean + 4.0*std)
    if clim is None:
        dbm = display_bscan.mean()
        dbstd = display_bscan.std()
        llim = dbm-0.5*dbstd
        ulim = dbm+4.0*dbstd
        clim = (llim,ulim)
    plt.imshow(display_bscan,clim=clim,cmap='gray',aspect='auto')
    plt.colorbar()
    plt.title(title)
    if plot_ascan:
        plt.figure()
        plt.plot(display_bscan.mean(1))
    #plt.savefig('./figs/%s.png'%(title.replace(' ','_')))

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

def bin_shift_histogram(vals,bin_centers,resample_factor=1,do_plots=False):
    
    shifts = np.linspace(bin_centers[0]/float(len(bin_centers)),
                          bin_centers[-1]/float(len(bin_centers)),resample_factor)

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

    if do_plots:
        bin_edges = centers_to_edges(bin_centers)
        bin_width = np.mean(np.diff(bin_edges))
        shift_size = np.mean(np.diff(shifts))
        
        plt.figure()
        plt.imshow(all_counts)
        plt.title('counts')
        plt.xlabel('bins')
        plt.ylabel('shifts')
        plt.gca().set_yticks(np.arange(n_shifts))
        plt.gca().set_yticklabels(['%0.1f'%s for s in shifts])
        plt.gca().set_xticks(np.arange(n_bins))
        plt.gca().set_xticklabels(['%0.1f'%bc for bc in bin_centers])
        plt.colorbar()

        plt.figure()
        plt.imshow(all_centers)
        plt.title('bin centers')
        plt.xlabel('bins')
        plt.ylabel('shifts')
        plt.gca().set_yticks(np.arange(n_shifts))
        plt.gca().set_yticklabels(['%0.1f'%s for s in shifts])
        plt.gca().set_xticks(np.arange(n_bins))
        plt.gca().set_xticklabels(['%0.1f'%bc for bc in bin_centers])
        plt.colorbar()

        all_counts = all_counts.T
        all_centers = all_centers.T.ravel()

        plt.figure()
        plt.subplot(2,1,1)
        plt.hist(vals,bin_edges,width=bin_width*0.8)
        plt.subplot(2,1,2)
        plt.bar(all_centers.ravel(),all_counts.ravel(),width=shift_size*0.8)
        plt.show()

    return all_counts.T.ravel(),all_centers.T.ravel()


def test_bin_shift_histogram():
    vals = np.random.randn(100)
    bin_edges = np.linspace(-4,4,9)
    resample_factor = 10
    counts,centers = bin_shift_histogram(vals,bin_edges,resample_factor,do_plots=True)
    print(counts,centers)
    sys.exit()
    
#test_bin_shift_histogram()

def wrap_into_range(arr,phase_limits=(-np.pi,np.pi)):
    lower,upper = phase_limits
    above_range = np.where(arr>upper)
    below_range = np.where(arr<lower)
    arr[above_range]-=2*np.pi
    arr[below_range]+=2*np.pi
    return arr


def bulk_motion_correct_original(phase_stack,mask,
                                 n_bins=pp.bulk_motion_n_bins,
                                 resample_factor=pp.bulk_motion_resample_factor,
                                 n_smooth=pp.bulk_motion_n_smooth):

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
    d_phase_d_t = np.transpose(np.transpose(d_phase_d_t,(2,0,1))*mask,(1,2,0))
    d_phase_d_t = wrap_into_range(d_phase_d_t)
    
    bin_edges = np.linspace(-np.pi,np.pi,n_bins)
    
    # The key idea here is from Makita, 2006, where it is well explained. In
    # addition to using the phase mode, we also do bin-shifting, in order to
    # smooth the histogram. Again departing from Justin's approach, let's
    # just specify the top level bins and a resampling factor, and let the
    # histogram function do all the work of setting the shifted bin edges.

    b_jumps = np.zeros((d_phase_d_t.shape[1:]))
    
    for f in range(n_fast):
        valid_idx = mask[:,f]
        for r in range(n_reps-1):
            vals = d_phase_d_t[valid_idx,f,r]
            [counts,bin_centers] = bin_shift_histogram(vals,bin_edges,resample_factor,do_plots=False)
            bulk_shift = bin_centers[np.argmax(counts)]
            b_jumps[f,r] = bulk_shift

    # Now unwrap to prevent discontinuities (although this may not impact complex variance)
    b_jumps = np.unwrap(b_jumps,axis=0)

    # Smooth by convolution. Don't forget to divide by kernel size!
    b_jumps = sps.convolve2d(b_jumps,np.ones((n_smooth,1)),mode='same')/float(n_smooth)

    # Now, subtract b_jumps from phase_stack, not including the first repeat
    # Important: this is happening by broadcasting--it requires that the
    # last two dimensions of phase_stack[:,:,1:] be equal in size to the two
    # dimensions of b_jumps
    out = np.copy(phase_stack)
    for rep in range(1,n_reps):
        # for each rep, the total error is the sum of
        # all previous errors
        err = np.sum(b_jumps[:,:rep],axis=1)
        out[:,:,rep] = out[:,:,rep]-err
        
    out = wrap_into_range(out)

    return out

def get_phase_jumps(phase_stack,mask,
                    n_bins=pp.bulk_motion_n_bins,
                    resample_factor=pp.bulk_motion_resample_factor,
                    n_smooth=pp.bulk_motion_n_smooth):

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
    d_phase_d_t = np.transpose(np.transpose(d_phase_d_t,(2,0,1))*mask,(1,2,0))
    d_phase_d_t = wrap_into_range(d_phase_d_t)
    
    bin_edges = np.linspace(-np.pi,np.pi,n_bins)
    
    # The key idea here is from Makita, 2006, where it is well explained. In
    # addition to using the phase mode, we also do bin-shifting, in order to
    # smooth the histogram. Again departing from Justin's approach, let's
    # just specify the top level bins and a resampling factor, and let the
    # histogram function do all the work of setting the shifted bin edges.

    b_jumps = np.zeros((d_phase_d_t.shape[1:]))
    
    for f in range(n_fast):
        valid_idx = mask[:,f]
        for r in range(n_reps-1):
            vals = d_phase_d_t[valid_idx,f,r]
            [counts,bin_centers] = bin_shift_histogram(vals,bin_edges,resample_factor,do_plots=False)
            bulk_shift = bin_centers[np.argmax(counts)]
            b_jumps[f,r] = bulk_shift

    # Now unwrap to prevent discontinuities (although this may not impact complex variance)
    b_jumps = np.unwrap(b_jumps,axis=0)

    # Smooth by convolution. Don't forget to divide by kernel size!
    b_jumps = sps.convolve2d(b_jumps,np.ones((n_smooth,1)),mode='same')/float(n_smooth)

    return b_jumps

def bulk_motion_correct(phase_stack,mask,
                        n_bins=pp.bulk_motion_n_bins,
                        resample_factor=pp.bulk_motion_resample_factor,
                        n_smooth=pp.bulk_motion_n_smooth):

    # Take a stack of B-scan phase arrays, with dimensions
    # (z,x,repeats), and return a bulk-motion corrected
    # version
    
    n_reps = phase_stack.shape[2]

    b_jumps = get_phase_jumps(phase_stack,mask,
                              n_bins=pp.bulk_motion_n_bins,
                              resample_factor=pp.bulk_motion_resample_factor,
                              n_smooth=pp.bulk_motion_n_smooth)

    # Now, subtract b_jumps from phase_stack, not including the first repeat
    # Important: this is happening by broadcasting--it requires that the
    # last two dimensions of phase_stack[:,:,1:] be equal in size to the two
    # dimensions of b_jumps
    out = np.copy(phase_stack)
    for rep in range(1,n_reps):
        # for each rep, the total error is the sum of
        # all previous errors
        err = np.sum(b_jumps[:,:rep],axis=1)
        out[:,:,rep] = out[:,:,rep]-err
        
    out = wrap_into_range(out)

    return out

def nancount(arr):
    return len(np.where(np.isnan(arr))[0])

def phase_variance(data_phase,mask):
    # Assumes the temporal dimension is the last, dim 2
    # ddof=1 means delta degrees of freedom = 1,
    # i.e. variance is computed with N-1 in denominator
    pv = np.var(np.exp(1j*data_phase),axis=2,ddof=1)
    pv = pv*mask
    pv[pv>1] = 1.0
    pv[pv<0] = 0.0
    return pv

def make_angiogram(stack_complex,bulk_correction_threshold=None,phase_variance_threshold=None,diagnostics=False):
    stack_amplitude = np.abs(stack_complex)
    stack_log_amplitude = 20*np.log10(stack_amplitude)
    stack_phase = np.angle(stack_complex)

    # Inferring this dB threshold from the number of pixels
    # in Justin's mask, since I'm going to skip all the confusing
    # scaling steps.
    # Update: Justin's scaling from dB into ADU is not linear,
    # so all bets are off about these dB thresholds working;
    # implementing Justin's non-linear scaling for now, but this
    # has to be gotten rid of eventually.
    maintain_db_units = False

    # if we wanted to maintain_db_units:
    #    bulk_correction_threshold = 56.40253 # dB, should give 83081 1's in mask
    #    phase_variance_threshold = 62.76747 # dB, 42488 1's in mask

    CSTD = np.std(np.mean(stack_log_amplitude,2))
    FMID = np.mean(np.mean(stack_log_amplitude,2))
    stack_log_amplitude = stack_log_amplitude-(FMID-0.9*CSTD)
    
    stack_log_amplitude = stack_log_amplitude/stack_log_amplitude.max()
    stack_amplitude = stack_amplitude/stack_amplitude.max()

    stack_log_amplitude[stack_log_amplitude<0] = 0.0

    if bulk_correction_threshold is None:
        bulk_correction_threshold = pp.bulk_correction_threshold

    if phase_variance_threshold is None:
        phase_variance_threshold = pp.phase_variance_threshold


    mean_log_amplitude_stack = np.mean(stack_log_amplitude,2)
    
    bulk_correction_mask = (mean_log_amplitude_stack>bulk_correction_threshold)
    phase_variance_mask = (mean_log_amplitude_stack>phase_variance_threshold)

    if diagnostics:
        plt.figure()
        plt.imshow(mean_log_amplitude_stack,cmap='gray',aspect='auto')
        plt.title('diagnostics: log b-scan')
        plt.figure()
        plt.imshow(bulk_correction_mask,cmap='gray',aspect='auto')
        plt.title('diagnostics: bulk correction mask')
        plt.figure()
        plt.imshow(phase_variance_mask,cmap='gray',aspect='auto')
        plt.title('diagnostics: phase variance mask')

    stack_phase = bulk_motion_correct(stack_phase,bulk_correction_mask)
    pv = phase_variance(stack_phase,phase_variance_mask)

    return pv



