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
from octoblob import registration_tools

IPSP = 4.0
DISPLAY_DPI = 75
PRINT_DPI = 75

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

        self.saturation_value = np.iinfo(self.dtype).max
        
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


    def align_to_fbg(self,frame,region_height=48,smoothing_size=5,sign=1,use_cross_correlation=False,diagnostics=False):
        # The algorithm here is copied from Justin Migacz's MATLAB prototype; one
        # key difference is that Justin cats 5 sets of spectra together, such that
        # they share the same k-axis; this step is performed on a "compound" frame,
        # with shape (n_k,n_fast*n_repeats)
        if not self.has_fbg:
            return frame
        z1 = self.fbg_position-region_height//2
        z2 = self.fbg_position+region_height//2

        # crop the relevant region:
        fbg_region = np.zeros((z2-z1,frame.shape[1]))
        fbg_region[:,:] = frame[z1:z2,:]


        if not use_cross_correlation:

            # smooth with a kernel of 1x2 pixels
            # use valid region (MATLAB default) to avoid huge gradients
            # at edges
            # RSJ 23 March 2021: smoothing here may create a problem, viz.
            # a +/- 1 pixel uncertainty about the location of the
            # FBG edge; this may have caused the high-frequency (A-scan rate)
            # movements of the bulk phase mode observed in the histograms.
            # Removing this line for now:
            # fbg_region = sps.convolve2d(fbg_region,np.ones((smoothing_size,1)),mode='valid')

            # do the vertical derivative:
            fbg_region_derivative = sign*np.diff(fbg_region,axis=0)

            # find the index of maximum rise for each column in the derivative:
            # the +1 at the end is just to square this with Justin's code;
            # ideally, get rid of the +1 here and add back the zero-centering below

            position = np.argsort(fbg_region_derivative,axis=0)[-1,:].astype(np.int)+1

            # zero-center the position to avoid rolling more than we need to;
            # this departs a bit from Justin's approach, but not consequentially
            # consider adding this back in after the code has been thoroughly checked
            # position = position - int(np.round(np.mean(position)))


        else:

            ftar = np.fft.fft(fbg_region,axis=0)
            fref = ftar[:,ftar.shape[1]//2]
            xc = np.abs(np.fft.ifft((ftar.T*np.conj(fref)).T,axis=0))
            
            position = np.argmax(xc,axis=0)
            position = position - int(np.round(np.mean(position)))


        posrms = position.std()
        # roll the columns to align
        # Notes:
        # 1. After testing, replace this with an in-place operation,
        #    e.g. frame[:,x] = np.roll(....
        # 2. Ideally, get rid of the loop--it's not necessary; use
        #    advanced indexing, but verify that it's CPU-vectorized,
        #    o.w. leave as is here for clarity

        out = np.zeros(frame.shape)
        for x in range(frame.shape[1]):
            out[:,x] = np.roll(frame[:,x],-position[x])
            
        if diagnostics:
            n_samples = 10
            sample_interval = int(float(out.shape[1])/float(n_samples))
            plt.figure(figsize=(4*IPSP,2*IPSP),dpi=DISPLAY_DPI)
            plt.subplot(2,5,1)
            plt.imshow(frame,cmap='gray',aspect='auto',interpolation='none')
            plt.axhline(z1,alpha=0.5)
            plt.axhline(z2,alpha=0.5)
            plt.title('uncorrected and search region')


            plt.subplot(2,5,2)
            plt.imshow(frame,cmap='gray',aspect='auto',interpolation='none')
            plt.axhline(z1,alpha=0.75,linewidth=2)
            plt.axhline(z2,alpha=0.75,linewidth=2)
            plt.ylim((z2+20,z1-20))
            plt.title('(zoomed)')

            
            plt.subplot(2,5,3)
            for f in range(0,frame.shape[1],sample_interval):
                plt.plot(frame[:,f],label='%d'%f)
            plt.legend(fontsize=6)
                
            plt.xlim((z1-10,z2+10))
            plt.axvline(z1,alpha=0.5)
            plt.axvline(z2,alpha=0.5)
            plt.title('sample uncorrected spectra')


            plt.subplot(2,5,4)
            plt.plot(frame.mean(1))
            plt.xlim((z1-10,z2+10))
            plt.axvline(z1,alpha=0.5)
            plt.axvline(z2,alpha=0.5)
            plt.title('uncorrected average')


            mid = frame.shape[1]//2
            freq = np.fft.fftfreq(frame.shape[1],1.0)[5:mid]
            spec = np.mean(np.abs(np.fft.fft(frame,axis=1)),axis=0)[5:mid]
            spec = spec**2
            plt.subplot(2,5,5)
            plt.semilogy(freq,spec)
            spec_ylim = plt.ylim()
            plt.title('modulation power spectrum')
            spec_ax = plt.gca()
            
            plt.subplot(2,5,6)
            plt.imshow(out,cmap='gray',aspect='auto',interpolation='none')
            plt.title('corrected ($\sigma_z=%0.1fpx$)'%posrms)

            plt.subplot(2,5,7)
            plt.imshow(out,cmap='gray',aspect='auto',interpolation='none')
            plt.ylim((z2+20,z1-20))
            plt.title('(zoomed)')

            plt.subplot(2,5,8)
            for f in range(0,out.shape[1],sample_interval):
                plt.plot(out[:,f],label='%d'%f)
            plt.legend(fontsize=6)
            plt.xlim((z1-10,z2+10))
            plt.title('sample corrected spectra')

            plt.subplot(2,5,9)
            plt.plot(out.mean(1))
            plt.xlim((z1-10,z2+10))
            plt.title('corrected average')

            outspec = np.mean(np.abs(np.fft.fft(out,axis=1)),axis=0)[5:mid]
            outspec = outspec**2
            plt.subplot(2,5,10)
            plt.semilogy(freq,outspec)
            outspec_ylim = plt.ylim()
            plt.title('corrected power spectrum')
            plt.xlabel('freq (x scan rate)')
            global_ylim = (min(spec_ylim[0],outspec_ylim[0]),max(spec_ylim[1],outspec_ylim[1]))
            plt.ylim(global_ylim)
            spec_ax.set_ylim(global_ylim)
            plt.suptitle('align_to_fbg diagnostics')
            save_diagnostics(diagnostics,'fbg_alignment')
        return out

    
    def get_frame(self,frame_index,volume_index=0,diagnostics=False):
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

            if frame.max()>=self.saturation_value:
                if diagnostics:
                    plt.figure(figsize=(IPSP,IPSP),dpi=DISPLAY_DPI)
                    plt.hist(frame,bins=100)
                    plt.title('Frame saturated with pixels >= %d.'%self.saturation_value)
                print('Frame saturated, with pixels >= %d.'%self.saturation_value)
            
            if diagnostics:
                plt.figure(figsize=(IPSP,2*IPSP),dpi=DISPLAY_DPI)
                plt.subplot(2,1,1)
                plt.hist(frame,bins=100)
                plt.title('before %d bit shift'%self.bit_shift_right)
                
            # Bit-shift if necessary, e.g. for Axsun data
            if self.bit_shift_right:
                frame = np.right_shift(frame,self.bit_shift_right)

            if diagnostics:
                plt.subplot(2,1,2)
                plt.hist(frame,bins=100)
                plt.title('after %d bit shift'%self.bit_shift_right)
                
                
            # Reshape into the k*x 2D array
            frame = frame.reshape(self.n_fast,self.n_depth).T


            if diagnostics:
                plt.figure(figsize=(IPSP,IPSP),dpi=DISPLAY_DPI)
                plt.imshow(frame,aspect='auto',interpolation='none',cmap='gray')
                plt.colorbar()
                plt.title('raw data (bit shifted %d bits)'%self.bit_shift_right)
                save_diagnostics(diagnostics,'raw_data')
            
            # If there's an fbg, align spectra using the align_to_fbg function
            if self.has_fbg:
                frame = self.align_to_fbg(frame,sign=self.fbg_sign,diagnostics=diagnostics)

            frame = frame[self.spectrum_start:self.spectrum_end,:]
        return frame


class Resampler:

    def __init__(self,lambda0,d_lambda,n_points):
        ##### Add possibility for remapping here; store it in the object so the interpolator
        ##### persists and we don't have to recalc the interpolator for every frame
        ##### (Actually this would be a good thing to do for k-remapping too)
        self.lambda0 = lambda0
        self.d_lambda = d_lambda
        self.n_points = n_points
        #self.wavelength_spectrum = np.polyval([4.1e-11,8.01e-7],np.arange(points_per_spectrum))
        self.wavelength_spectrum = np.polyval([self.d_lambda,self.lambda0],np.arange(self.n_points))
        self.k_in = 2.0*np.pi/self.wavelength_spectrum
        self.k_out = np.linspace(self.k_in[0],self.k_in[-1],self.n_points)

    def map(self,spectra):
        k_interpolator = spi.interp1d(self.k_in,spectra,axis=0,copy=False)
        return k_interpolator(self.k_out)


def save_diagnostics(diagnostics,tag):
    # if diagnostics is a tuple containing a directory and a frame index,
    # save them to a subdirectory, named by index
    # if not, return
    try:
        directory = diagnostics[0]
        index = diagnostics[1]
        subdir = os.path.join(directory,tag)
        os.makedirs(subdir,exist_ok=True)
    except:
        return
    
    try:
        plt.savefig(os.path.join(subdir,'%05d.png'%index),dpi=PRINT_DPI)
    except Exception as e:
        print('save_diagnostics error: %s'%e)

def log_diagnostics(diagnostics,tag,header,data,fmt=None,clobber=False):
    # if diagnostics is a tuple containing a directory and a frame index,
    # save them to a subdirectory, named by index
    # if not, return
    try:
        directory = diagnostics[0]
        index = diagnostics[1]
        subdir = os.path.join(directory,tag)
        os.makedirs(subdir,exist_ok=True)
    except:
        return

    try:
        assert len(header)==len(data)
    except AssertionError:
        print('log_diagnostics received header and data of different lengths')
        print('header:')
        print(header)
        print('data:')
        print(data)

    if fmt is None:
        fmt = ['%0.3f']*len(header)
    elif len(fmt)==1:
        fmt = fmt*len(header)
    else:
        try:
            assert len(fmt)==len(data)
        except AssertionError:
            print('log_diagnostics received fmt and data of different lengths')
            print('fmt:')
            print(fmt)
            print('data:')
            print(data)
        
    try:
        log_fn = os.path.join(subdir,'%05d.csv'%index)
        data_string = ','.join([f%d for f,d in zip(fmt,data)])+'\n'
        if os.path.exists(log_fn) and not clobber:
            with open(log_fn,'a') as fid:
                fid.write(data_string)
        else:
            header_string = ','.join([h for h in header])+'\n'
            with open(log_fn,'w') as fid:
                fid.write(header_string)
                fid.write(data_string)
    except Exception as e:
        print('log_diagnostics error: %s'%e)

    
def dc_subtract(spectra,diagnostics=False):
    """Estimate DC by averaging spectra spatially (dimension 1),
    then subtract by broadcasting."""
    dc = spectra.mean(1)
    out = (spectra.T-dc).T
    
    if diagnostics:
        plt.figure(figsize=(2*IPSP,2*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(2,2,1)
        plt.imshow(spectra,aspect='auto',cmap='gray',interpolation='none')
        plt.colorbar()
        plt.title('uncorrected spectra')
        plt.subplot(2,2,2)
        plt.plot(dc)
        plt.xlabel('spectral index')
        plt.ylabel('amplitude')
        plt.title('estimated DC')
        plt.subplot(2,2,3)
        plt.imshow(out,aspect='auto',cmap='gray',interpolation='none')
        plt.title('DC corrected spectra')
        plt.colorbar()
        plt.subplot(2,2,4)
        plt.hist(np.ravel(out),bins=range(-1000,1010,10))
        plt.xlabel('DC corrected amplitude')
        plt.ylabel('count')
        save_diagnostics(diagnostics,'dc_subtraction')
    return out


def k_resample(spectra,coefficients=pp.k_resampling_coefficients,diagnostics=False):
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

    if diagnostics:
        plt.figure(figsize=(2*IPSP,3*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(3,2,1)
        plt.imshow(spectra)
        plt.colorbar()
        plt.title('spectra before mapping')
        plt.subplot(3,2,2)
        plt.imshow(interpolated)
        plt.colorbar()
        plt.title('spectra after mapping')
        plt.subplot(3,2,3)
        plt.imshow(np.log(np.abs(np.fft.fft(spectra,axis=0))))
        plt.colorbar()
        plt.title('fft before mapping')
        plt.subplot(3,2,4)
        plt.imshow(np.log(np.abs(np.fft.fft(interpolated,axis=0))))
        plt.colorbar()
        plt.title('fft after mapping')
        plt.subplot(3,2,5)
        plt.hist(np.ravel(np.log(np.abs(np.fft.fft(spectra,axis=0)))),bins=100)
        plt.colorbar()
        plt.title('fft before mapping')
        plt.subplot(3,2,6)
        plt.hist(np.ravel(np.log(np.abs(np.fft.fft(interpolated,axis=0)))),bins=100)
        plt.colorbar()
        plt.title('fft after mapping')

        
        plt.suptitle('k_resample diagnostics')
        
    
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
        plt.figure(figsize=(2*IPSP,1*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(1,2,1)
        plt.imshow(before,cmap='gray',aspect='auto',clim=[40,80])
        plt.colorbar()
        plt.title('before disp. comp. (dB)')
        plt.subplot(1,2,2)
        plt.imshow(after,cmap='gray',aspect='auto',clim=[40,80])
        plt.colorbar()
        plt.title('after disp. comp. (dB)')
        save_diagnostics(diagnostics,'dispersion_compensation')
    return dechirped
    

def gaussian_window(spectra,sigma=pp.gaussian_window_sigma,diagnostics=False):
    # WindowMat = repmat(exp(-((linspace(-1,1,size(Aspectra,1)))'.^2)/SIG),[1,C*D2]);
    x = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma))
    out = (spectra.T*x).T
    if diagnostics:
        def get_dc(frame):
            #frame = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(frame,axis=0),axes=(0))))
            frame = np.abs(np.fft.fftshift(np.fft.fft(frame,axis=0),axes=(0)))
            sy,sx = frame.shape
            dcz1 = sy//2-10
            dcz2 = sy//2+10
            dc = frame[dcz1:dcz2,:]
            vprof = np.mean(frame,axis=1)
            vprof[dcz1:dcz2] = 0
            bind = np.argmax(vprof)
            bz1 = max(bind-10,0)
            bz2 = min(bind+10,sy)
            bright = frame[bz1:bz2,:]
            return dc,bright

            
        plt.figure(figsize=(2*IPSP,4*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(4,2,1)
        plt.imshow(np.abs(spectra),cmap='gray',aspect='auto',interpolation='none')
        plt.colorbar()
        plt.title('amplitude of complex fringe\nbefore Gaussian windowing')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4,2,2)
        plt.imshow(np.abs(out),cmap='gray',aspect='auto',interpolation='none')
        plt.colorbar()
        plt.title('amplitude of complex fringe\nafter Gaussian windowing')
        plt.xticks([])
        plt.yticks([])


        predc,prebright = get_dc(spectra)
        postdc,postbright = get_dc(out)
        
        plt.subplot(4,2,3)
        plt.imshow(predc,cmap='gray',aspect='auto',interpolation='none',clim=np.percentile(predc,(.5,99.5)))
        plt.colorbar()
        plt.title('DC before windowing (dB)')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(4,2,4)
        plt.imshow(postdc,cmap='gray',aspect='auto',interpolation='none',clim=np.percentile(predc,(.5,99.5)))
        plt.colorbar()
        plt.title('DC after windowing (dB)')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,2,5)
        plt.imshow(prebright,cmap='gray',aspect='auto',interpolation='none',clim=np.percentile(prebright,(.5,99.5)))
        plt.colorbar()
        plt.title('bright band before windowing (dB)')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(4,2,6)
        plt.imshow(postbright,cmap='gray',aspect='auto',interpolation='none',clim=np.percentile(postbright,(.5,99.5)))
        plt.colorbar()
        plt.title('bright band after windowing (dB)')
        plt.xticks([])
        plt.yticks([])

        
        plt.subplot(4,2,7)
        plt.plot(x)
        plt.title('Gaussian window')
        
    return out

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
        plt.figure(figsize=(1*IPSP,1*IPSP),dpi=DISPLAY_DPI)
        plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray',clim=[40,80],aspect='auto')
        plt.colorbar()
        plt.axhline(z1)
        if z2>=0:
            plt.axhline(z2)
        else:
            plt.axhline(bscan.shape[0]+z2)
        plt.title('diagnostics: cropped region, contrast limited to (40,80) dB')
        save_diagnostics(diagnostics,'cropped_region')
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
        plt.figure(figsize=(1*IPSP,1*IPSP),dpi=DISPLAY_DPI)
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

def bin_shift_histogram(vals,bin_centers,resample_factor=1,diagnostics=False):
    shifts = np.linspace(bin_centers[0]/float(len(bin_centers)),
                          bin_centers[-1]/float(len(bin_centers)),resample_factor)

    print('shifts:')
    print(shifts)

    print('bin centers:')
    print(bin_centers)
    
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
        
        plt.figure(figsize=(3*IPSP,IPSP),dpi=DISPLAY_DPI)
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
        save_diagnostics(diagnostics,'bin_shift_histogram')

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


def get_phase_jumps(phase_stack,mask,
                    n_bins=pp.bulk_motion_n_bins,
                    resample_factor=pp.bulk_motion_resample_factor,
                    n_smooth=pp.bulk_motion_n_smooth,polynomial_smoothing=True,diagnostics=False):

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
        plt.figure(figsize=((n_reps-1)*IPSP,2*IPSP),dpi=DISPLAY_DPI)
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
        plt.figure(figsize=((n_reps-1)*IPSP,1*IPSP),dpi=DISPLAY_DPI)
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
        fig = plt.figure(figsize=(IPSP*4,IPSP*4))
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
        fig = plt.figure(figsize=(IPSP*4,IPSP*4))
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



def polynomial_smooth_phase(counts,shifts):
    import scipy.optimize as spo
    n_reps = shifts.shape[1]
    t = np.arange(shifts.shape[0])

    def objective_helper(complex_data,coefs,weights=None):
        x = np.arange(len(data))
        phase_fit = np.polyval(coefs,x)
        complex_fit = 1.0*np.exp(-1j*phase_fit)
        sqerr = np.abs((complex_data-complex_fit)**2)
        if weights is not None:
            sqerr = sqerr*weights
        out = np.sqrt(np.sum(sqerr))
        print(out)
        return out

        
    for rep in range(n_reps):
        data = shifts[:,rep]
        complex_data = 1.0*np.exp(-1j*data)
        objective = lambda coefs: objective_helper(complex_data,coefs,weights=counts[:,rep]**0.5)
        coefs0 = [0.0,0.0]
        res = spo.minimize(objective,coefs0)
        coefs = res.x

        plt.plot(t,np.real(complex_data),'k.')
        plt.plot(t,np.real(1.0*np.exp(-1j*np.polyval(coefs,t))),'b-')
        plt.show()
        print(res)
        sys.exit()

def polynomial_smooth_phase0(counts,shifts):
    import scipy.optimize as spo
    
    t = np.arange(shifts.shape[0])
    def objective_helper(data,coefs,weights=None):
        x = np.arange(len(data))
        order = len(coefs)-1
        fit = np.polyval(coefs,x)
        fit = (fit%(np.pi*2))-np.pi
        sqerr = ((x-fit)**2)
        if weights is not None:
            sqerr = sqerr*weights
        out = np.sqrt(np.sum(sqerr))
        print(out)
        return out
        
    n_reps = shifts.shape[1]

    for rep in range(n_reps):
        data = shifts[:,rep]
        weights = counts[:,rep] # consider squaring this
        objective = lambda coefs: objective_helper(data,coefs,None)
        x0 = [-1.0/25.0,-50.0]
        res = spo.minimize(objective,x0)
        print(res.x)

        
        manual_fit = np.polyval(x0,t)%(2*np.pi)-np.pi
        auto_fit = np.polyval(res.x,t)%(2*np.pi)-np.pi
        plt.plot(t,data,'ks')
        plt.plot(t,manual_fit,'b-')
        plt.plot(t,auto_fit,'r-')
        plt.show()
        print(res)
        sys.exit()

def bulk_motion_correct(phase_stack,mask,
                        n_bins=pp.bulk_motion_n_bins,
                        resample_factor=pp.bulk_motion_resample_factor,
                        n_smooth=pp.bulk_motion_n_smooth,diagnostics=False):

    # Take a stack of B-scan phase arrays, with dimensions
    # (z,x,repeats), and return a bulk-motion corrected
    # version

    n_reps = phase_stack.shape[2]

    b_jumps = get_phase_jumps(phase_stack,mask,
                              n_bins=n_bins,
                              resample_factor=resample_factor,
                              n_smooth=n_smooth,
                              diagnostics=diagnostics)

    # Now, subtract b_jumps from phase_stack, not including the first repeat
    # Important: this is happening by broadcasting--it requires that the
    # last two dimensions of phase_stack[:,:,1:] be equal in size to the two
    # dimensions of b_jumps
    out = np.copy(phase_stack)

    if diagnostics:
        #err_clim = (np.min(np.sum(b_jumps,axis=1)),np.max(np.sum(b_jumps,axis=1)))
        phase_clim = (-np.pi,np.pi)
        err_clim = [-np.pi-np.min(-np.sum(b_jumps,axis=1)),np.pi+np.max(-np.sum(b_jumps,axis=1))]
        if err_clim[1]<err_clim[0]:
            err_clim = [-ec for ec in err_clim]
        plt.figure(figsize=((n_reps-1)*IPSP,2*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(2,n_reps+1,1)
        plt.imshow(mask*phase_stack[:,:,0],clim=phase_clim,aspect='auto',interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('frame 0')
        plt.ylabel('before correction')
        
        plt.subplot(2,n_reps+1,n_reps+2)
        plt.imshow(mask*out[:,:,0],clim=err_clim,aspect='auto',interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('frame 0')
        plt.ylabel('after correction')

    errs = []
    for rep in range(1,n_reps):
        # for each rep, the total error is the sum of
        # all previous errors
        err = np.sum(b_jumps[:,:rep],axis=1)
        errs.append(err)
        out[:,:,rep] = out[:,:,rep]-err
        if diagnostics:
            plt.subplot(2,n_reps+1,rep+1)
            plt.imshow(mask*phase_stack[:,:,rep],clim=phase_clim,aspect='auto',interpolation='none')
            plt.xlabel('frame %d'%rep)
            plt.xticks([])
            plt.yticks([])
            if rep==n_reps-1:
                plt.colorbar()

            plt.subplot(2,n_reps+1,n_reps+rep+2)
            plt.imshow(mask*out[:,:,rep],clim=err_clim,aspect='auto',interpolation='none')
            plt.xlabel('frame %d'%rep)
            plt.xticks([])
            plt.yticks([])
            if rep==n_reps-1:
                plt.colorbar()

    if diagnostics:
        plt.subplot(2,n_reps+1,n_reps+1)
        for idx,err in enumerate(errs):
            plt.plot(err,label='f%d'%(idx+1))
        plt.legend()
        save_diagnostics(diagnostics,'bulk_motion_correction')
        
    out = wrap_into_range(out)

    return out

def nancount(arr):
    return len(np.where(np.isnan(arr))[0])

def phase_variance(data_phase,mask,diagnostics=False):
    # Assumes the temporal dimension is the last, dim 2
    # ddof=1 means delta degrees of freedom = 1,
    # i.e. variance is computed with N-1 in denominator
    pv = np.var(np.exp(1j*data_phase),axis=2,ddof=1)
    if diagnostics:
        plt.figure(figsize=(3*IPSP,1*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(1,3,1)
        plt.imshow(pv,cmap='gray',aspect='auto',interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('PV before mask')
        plt.colorbar()
        
    pv = pv*mask
    if diagnostics:
        plt.subplot(1,3,2)
        plt.imshow(pv,cmap='gray',aspect='auto',interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('PV masked')
        plt.colorbar()
        
    pv[pv>1] = 1.0
    pv[pv<0] = 0.0
    if diagnostics:
        plt.subplot(1,3,3)
        plt.imshow(pv,cmap='gray',aspect='auto',interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('PV masked and clipped to [0,1]')
        plt.colorbar()
        save_diagnostics(diagnostics,'phase_variance')
        
    return pv

def make_angiogram(stack_complex,bulk_correction_threshold=None,phase_variance_threshold=None,diagnostics=False):
    stack_amplitude = np.abs(stack_complex)
    stack_dB = 20*np.log10(stack_amplitude)
    stack_phase = np.angle(stack_complex)

    mean_dB_original = np.mean(stack_dB,2)
    
    # Inferring this dB threshold from the number of pixels
    # in Justin's mask, since I'm going to skip all the confusing
    # scaling steps.
    # Update: Justin's scaling from dB into ADU is not linear,
    # so all bets are off about these dB thresholds working;
    # implementing Justin's non-linear scaling for now, but this
    # has to be gotten rid of eventually.
    maintain_dB_units = False

    # if we wanted to maintain_dB_units:
    #    bulk_correction_threshold = 56.40253 # dB, should give 83081 1's in mask
    #    phase_variance_threshold = 62.76747 # dB, 42488 1's in mask

    CSTD = np.std(np.mean(stack_dB,2))
    FMID = np.mean(np.mean(stack_dB,2))

    if diagnostics:
        nbins=20
        plt.figure(figsize=(4*IPSP,2*IPSP),dpi=DISPLAY_DPI)
        plt.subplot(2,4,1)
        plt.hist(np.ravel(stack_dB),bins=nbins)
        plt.xlabel('dB (full stack)')
        plt.axvline(FMID,color='r',label='dB mean')
        plt.axvspan(FMID,FMID+CSTD,color='m',alpha=0.2)
        plt.axvspan(FMID-CSTD,FMID,color='y',alpha=0.2)
        plt.legend()
        plt.yticks([])
        
    stack_dB = stack_dB-(FMID-0.9*CSTD)
    
    if diagnostics:
        plt.subplot(2,4,2)
        plt.hist(np.ravel(stack_dB),bins=nbins)
        plt.xlabel('centered (dB - mean(dB) + 0.9 std(dB))')
        plt.yticks([])
        
    stack_dB = stack_dB/stack_dB.max()
    
    if diagnostics:
        plt.subplot(2,4,3)
        plt.hist(np.ravel(stack_dB),bins=nbins)
        plt.xlabel('normalized by max')
        plt.yticks([])

    stack_amplitude = stack_amplitude/stack_amplitude.max()

    stack_dB[stack_dB<0] = 0.0
    
    if diagnostics:
        plt.subplot(2,4,4)
        out = plt.hist(np.ravel(stack_dB),bins=nbins)
        ymin1 = 0.2
        ymax1 = 0.45
        ymin2 = 0.5
        ymax2 = 0.75
        plt.xlabel('clipped at 0.0')
        plt.axvline(bulk_correction_threshold,ymin=ymin1,ymax=ymax1,color='g',label='bulk thresh')
        plt.axvline(phase_variance_threshold,ymin=ymin2,ymax=ymax2,color='r',label='pv thresh')
        plt.legend()
        plt.yticks([])
        
    if bulk_correction_threshold is None:
        bulk_correction_threshold = pp.bulk_correction_threshold

    if phase_variance_threshold is None:
        phase_variance_threshold = pp.phase_variance_threshold

    mean_dB_stack = np.mean(stack_dB,2)

    if diagnostics:
        plt.subplot(2,4,5)
        plt.imshow(mean_dB_original,cmap='gray',aspect='auto')
        plt.colorbar()
        plt.xlabel('original dB')
        
        plt.subplot(2,4,6)
        plt.imshow(mean_dB_stack,cmap='gray',aspect='auto')
        plt.colorbar()
        plt.xlabel('centered, normalized, clipped dB')
        plt.yticks([])
        
    bulk_correction_mask = (mean_dB_stack>bulk_correction_threshold)
    phase_variance_mask = (mean_dB_stack>phase_variance_threshold)

    if diagnostics:
        plt.subplot(2,4,7)
        plt.imshow(bulk_correction_mask,cmap='gray',aspect='auto')
        plt.xlabel('bulk correction mask')
        plt.yticks([])
        plt.subplot(2,4,8)
        plt.imshow(phase_variance_mask,cmap='gray',aspect='auto')
        plt.xlabel('phase variance mask')
        plt.yticks([])
        plt.suptitle('diagnostics: generation of bulk and pv masks')
        save_diagnostics(diagnostics,'phase_masks')

    stack_phase = bulk_motion_correct(stack_phase,bulk_correction_mask,diagnostics=diagnostics)
    pv = phase_variance(stack_phase,phase_variance_mask,diagnostics=diagnostics)

    return pv



# def bulk_motion_correct_original(phase_stack,mask,
#                                  n_bins=pp.bulk_motion_n_bins,
#                                  resample_factor=pp.bulk_motion_resample_factor,
#                                  n_smooth=pp.bulk_motion_n_smooth):

#     # Take a stack of B-scan phase arrays, with dimensions
#     # (z,x,repeats), and return a bulk-motion corrected
#     # version

#     n_depth = phase_stack.shape[0]
#     n_fast = phase_stack.shape[1]
#     n_reps = phase_stack.shape[2]
    
#     d_phase_d_t = np.diff(phase_stack,axis=2)

#     # multiply each frame of the diff array by
#     # the mask, so that only valid values remain;
#     # Then wrap any values above pi or below -pi into (-pi,pi) interval.
#     d_phase_d_t = np.transpose(np.transpose(d_phase_d_t,(2,0,1))*mask,(1,2,0))
#     d_phase_d_t = wrap_into_range(d_phase_d_t)
    
#     bin_edges = np.linspace(-np.pi,np.pi,n_bins)
    
#     # The key idea here is from Makita, 2006, where it is well explained. In
#     # addition to using the phase mode, we also do bin-shifting, in order to
#     # smooth the histogram. Again departing from Justin's approach, let's
#     # just specify the top level bins and a resampling factor, and let the
#     # histogram function do all the work of setting the shifted bin edges.

#     b_jumps = np.zeros((d_phase_d_t.shape[1:]))
    
#     for f in range(n_fast):
#         valid_idx = mask[:,f]
#         for r in range(n_reps-1):
#             vals = d_phase_d_t[valid_idx,f,r]
#             [counts,bin_centers] = bin_shift_histogram(vals,bin_edges,resample_factor,do_plots=True)
#             bulk_shift = bin_centers[np.argmax(counts)]
#             b_jumps[f,r] = bulk_shift

#     # Now unwrap to prevent discontinuities (although this may not impact complex variance)
#     b_jumps = np.unwrap(b_jumps,axis=0)

#     # Smooth by convolution. Don't forget to divide by kernel size!
#     b_jumps = sps.convolve2d(b_jumps,np.ones((n_smooth,1)),mode='same')/float(n_smooth)

#     # Now, subtract b_jumps from phase_stack, not including the first repeat
#     # Important: this is happening by broadcasting--it requires that the
#     # last two dimensions of phase_stack[:,:,1:] be equal in size to the two
#     # dimensions of b_jumps
#     out = np.copy(phase_stack)
#     for rep in range(1,n_reps):
#         # for each rep, the total error is the sum of
#         # all previous errors
#         err = np.sum(b_jumps[:,:rep],axis=1)
#         out[:,:,rep] = out[:,:,rep]-err
        
#     out = wrap_into_range(out)

#     return out

