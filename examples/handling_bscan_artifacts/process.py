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

# This example illustrates how to process a dataset that has artifacts such as lens reflexes and/or laser noise.
# The two key steps are 1) avoiding automatic cropping, which depends on the center of mass of the structure in
# the image (i.e., the location of the retina) to determine reasonable cropping points in depth (z); 2) forcing
# some horizontal cropping to remove the bright artifacts from the image; this latter step is critical for automatic
# optimization of dispersion/mapping coefficients.

# If we want automatic cropping (useful in most contexts) we use the function blobf.spectra_to_bscan, but in this case
# the artifact dominates the B-scan's center of mass and we have to use the full depth of the B-scan for optimization,
# thus we only use blobf.spectra_to_bscan_nocrop

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

# Get a plain frame for viewing
src = blobf.get_source(data_filename)
f = src.get_frame(10)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(f,aspect='auto')
plt.title('raw spectral frame')
plt.subplot(1,2,2)
plt.imshow(np.log10(np.abs(np.fft.fft(f,axis=0))),aspect='auto')
plt.title('FFT in k dimension')
os.makedirs('./figs',exist_ok=True)
plt.savefig('figs/artifact_example.png')


# After looking at the artifact_example.png, we can see that the first 200 columns of the image should be used for
# dispersion optimization, while skipping the rest of the image. Thus we'll re-create our source passing x1 and x2
# parameters of 0 and 200.

# Get an octoblob.DataSource object using the filename
src = blobf.get_source(data_filename,x1=0,x2=200)

# try to read dispersion/mapping coefs from a local processing_parameters file, and run optimization otherwise
try:
    coefs = np.array(params['mapping_dispersion_coefficients'],dtype=np.float)
    logging.info('File %s mapping dispersion coefficients found in %s. Skipping optimization.'%(data_filename,params_filename))
except KeyError:
    logging.info('File %s mapping dispersion coefficients not found in %s. Running optimization.'%(data_filename,params_filename))
    samples = src.get_samples(5)
    coefs = mdo.multi_optimize(samples,blobf.spectra_to_bscan_nocrop,show_all=False,show_final=True,verbose=False,diagnostics=diagnostics)
    params['mapping_dispersion_coefficients'] = coefs

# get the folder name for storing bscans
bscan_folder = file_manager.get_bscan_folder(data_filename)

if __name__=='__main__':

    if use_multiprocessing:
        def proc(k):
            # compute the B-scan from the spectra, using the provided dispersion coefficients:
            bscan = blobf.spectra_to_bscan_nocrop(coefs,src.get_frame(k),diagnostics=diagnostics)

            # save the complex B-scan in the B-scan folder
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)

        pool = mp.Pool(n_cores)
        pool.map(proc,range(src.n_total_frames))

    else:

        for k in range(src.n_total_frames):

            # compute the B-scan from the spectra, using the provided dispersion coefficients:
            bscan = blobf.spectra_to_bscan_nocrop(coefs,src.get_frame(k),diagnostics=diagnostics)

            # save the complex B-scan in the B-scan folder
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)

