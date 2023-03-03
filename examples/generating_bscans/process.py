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
use_multiprocessing = False
try:
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

# Get an octoblob.DataSource object using the filename
src = blobf.get_source(data_filename)

# try to read dispersion/mapping coefs from a local processing_parameters file, and run optimization otherwise
try:
    coefs = np.array(params['mapping_dispersion_coefficients'],dtype=np.float)
    logging.info('File %s mapping dispersion coefficients found in %s. Skipping optimization.'%(data_filename,params_filename))
except KeyError:
    logging.info('File %s mapping dispersion coefficients not found in %s. Running optimization.'%(data_filename,params_filename))
    samples = src.get_samples(5)
    coefs = mdo.multi_optimize(samples,blobf.spectra_to_bscan,show_all=False,show_final=True,verbose=False,diagnostics=diagnostics)
    params['mapping_dispersion_coefficients'] = coefs

# get the folder name for storing bscans
bscan_folder = file_manager.get_bscan_folder(data_filename)

if use_multiprocessing:
    def proc(k):
        # compute the B-scan from the spectra, using the provided dispersion coefficients:
        bscan = blobf.spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)

        # save the complex B-scan in the B-scan folder
        outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
        np.save(outfn,bscan)
        logging.info('Saving bscan %s.'%outfn)
        
    pool = mp.Pool(n_cores)
    pool.map(proc,range(src.n_total_frames))

else:

    for k in range(src.n_total_frames):

        # compute the B-scan from the spectra, using the provided dispersion coefficients:
        bscan = blobf.spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)

        # save the complex B-scan in the B-scan folder
        outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
        np.save(outfn,bscan)
        logging.info('Saving bscan %s.'%outfn)

