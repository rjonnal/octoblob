from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics_tools
from octoblob import parameters
from octoblob import org_tools
import sys,os,glob
import numpy as np
from octoblob import mapping_dispersion_optimizer as mdo
from octoblob import file_manager
import pathlib


# This example provides a new FBG alignment function based on cross-correlation. It may prove to be more robust than
# previous methods that used 'feature'-based alignment (e.g. aligning to the largest positive or negative gradients)

no_parallel = True

use_multiprocessing = False
try:
    assert not no_parallel
    import multiprocessing as mp
    use_multiprocessing = True
    n_cores_available = mp.cpu_count()
    n_cores = n_cores_available-2
    logging.info('multiprocessing imported')
    logging.info('n_cores_available: %d'%n_cores_available)
    logging.info('n_cores to be used: %d'%n_cores)
except ImportError as ie:
    logging.info('Failed to import multiprocessing: %s'%ie)
    logging.info('Processing serially.')
except AssertionError as ae:
    logging.info('Multiprocessing banned by no_parallel.')
    
data_filename = None

if data_filename is None:
    try:
        data_filename = sys.argv[1]
    except IndexError as ie:
        sys.exit('Please check data_filename. %s not found or data_filename not passed at command line.'%data_filename)


# Create a diagnostics object for inspecting intermediate processing steps
diagnostics = diagnostics_tools.Diagnostics(data_filename)

# Create a parameters object for storing and loading processing parameters
params_filename = file_manager.get_params_filename(data_filename)
params = parameters.Parameters(params_filename,verbose=True)


# New prototype fbg_align function, which uses cross-correlation instead of feature-
# based alignment of spectra.
# Set a limit on the maximum index where the FBG trough could possibly be located.
# This is a critical parameter, as it avoids cross correlation of spectra based on
# structural information; this would prevent the FBG features from dominating the
# cross-correlation and introduce additional phase noise.
# Correlation threshold is the minimum correlation required to consider two spectra
# to be in phase with one another
def fbg_align(spectra,fbg_max_index=150,correlation_threshold=0.9,diagnostics=None):
    # crop the frame to the FBG region
    f = spectra[:fbg_max_index,:].copy()

    if not diagnostics is None:
        fig = diagnostics.figure(figsize=(6,4))
        axes = fig.subplots(2,2)
        axes[0][0].imshow(f,aspect='auto')
        for k in range(f.shape[1]):
            axes[0][1].plot(f[:,k])

    # group the spectra by amount of shift
    # this step avoids having to perform cross-correlation operations on every
    # spectrum; first, we group them by correlation with one another
    # make a list of spectra to group
    to_do = list(range(f.shape[1]))
    # make a list for the groups of similarly shifted spectra
    groups = []
    ref = 0

    # while there are spectra left to group, do the following loop:
    while(True):
        groups.append([ref])
        to_do.remove(ref)
        for tar in to_do:
            c = np.corrcoef(f[:,ref],f[:,tar])[0,1]
            if c>correlation_threshold:
                groups[-1].append(tar)
                to_do.remove(tar)
        if len(to_do)==0:
            break
        ref = to_do[0]

    subframes = []
    for g in groups:
        subf = f[:,g]
        subframes.append(subf)

    # now decide how to shift the groups of spectra by cross-correlating their means
    # we'll use the first group as the reference group:
    group_shifts = [0]
    ref = np.mean(subframes[0],axis=1)
    # now, iterate through the other groups, compute their means, and cross-correlate
    # with the reference. keep track of the cross-correlation peaks in the list group_shifts
    for taridx in range(1,len(subframes)):
        tar = np.mean(subframes[taridx],axis=1)
        xc = np.fft.ifft(np.fft.fft(ref)*np.fft.fft(tar).conj())
        shift = np.argmax(xc)
        if shift>len(xc)//2:
            shift = shift-len(xc)
        group_shifts.append(shift)

    # now, use the groups and the group_shifts to shift all of the spectra according to their
    # group membership:
    for g,s in zip(groups,group_shifts):
        for idx in g:
            spectra[:,idx] = np.roll(spectra[:,idx],s)
            f[:,idx] = np.roll(f[:,idx],s)

    if not diagnostics is None:
        axes[1][0].imshow(f,aspect='auto')
        for k in range(f.shape[1]):
            axes[1][1].plot(f[:,k])
        diagnostics.save(fig)

    return spectra


def spectra_to_bscan(mdcoefs,spectra,diagnostics=None):
    # only the fbg_align function is called locally (from this script);
    # most of the OCT processing is done by blob functions (blobf.XXXX)
    spectra = fbg_align(spectra,diagnostics=diagnostics)
    spectra = blobf.dc_subtract(spectra,diagnostics=diagnostics)
    spectra = blobf.crop_spectra(spectra,diagnostics=diagnostics)
    spectra = blobf.k_resample(spectra,mdcoefs[:2],diagnostics=diagnostics)
    spectra = blobf.dispersion_compensate(spectra,mdcoefs[2:],diagnostics=None)
    spectra = blobf.gaussian_window(spectra,sigma=0.9,diagnostics=None)

    # Now generate the bscan by FFT:
    bscan = np.fft.fft(spectra,axis=0)
    # remove the upper half of the B-scan and leave only the bottom half:
    bscan = bscan[bscan.shape[0]//2:,:]

    # could additionally crop the B-scan if desired;
    # for example, could remove the top 10 rows, bottom 50 rows, and 10 columns
    # from the left and right edges:
    # bscan = bscan[10:-50,10:-10]

    # artifact.png has a lens flare artifact after the 150th column, so we'll remove
    # it; we'll also remove 50 rows near the DC (bottom of the image):
    bscan = bscan[:-50,:150]
    
    if not diagnostics is None:
        fig = diagnostics.figure()
        axes = fig.subplots(1,1)
        im = axes.imshow(20*np.log10(np.abs(bscan)),aspect='auto')
        plt.colorbar(im)
        diagnostics.save(fig)
    return bscan


# try to read dispersion/mapping coefs from a local processing_parameters file, and run optimization otherwise
try:
    coefs = np.array(params['mapping_dispersion_coefficients'],dtype=np.float)
    logging.info('File %s mapping dispersion coefficients found in %s. Skipping optimization.'%(data_filename,params_filename))
except KeyError:
    logging.info('File %s mapping dispersion coefficients not found in %s. Running optimization.'%(data_filename,params_filename))
    samples = src.get_samples(3)
    # modify the next line to use the local spectra_to_bscan function by removing 'blobf.':
    coefs = mdo.multi_optimize(samples,spectra_to_bscan,show_all=False,show_final=True,verbose=False,diagnostics=diagnostics)
    params['mapping_dispersion_coefficients'] = coefs

# get the folder name for storing bscans
bscan_folder = file_manager.get_bscan_folder(data_filename)

if __name__=='__main__':

    if use_multiprocessing:
        def proc(k):
            # compute the B-scan from the spectra, using the provided dispersion coefficients;
            # use the local spectra_to_bscan function, not the blobf. version
            bscan = spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)

            # save the complex B-scan in the B-scan folder
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)

        pool = mp.Pool(n_cores)
        pool.map(proc,range(src.n_total_frames))

    else:

        for k in range(src.n_total_frames):

            # compute the B-scan from the spectra, using the provided dispersion coefficients:
            # use the local spectra_to_bscan function, not the blobf. version
            bscan = spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)

            # save the complex B-scan in the B-scan folder
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)

