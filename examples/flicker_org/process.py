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

def process_dataset(data_filename):
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

    
    
    while src.has_more_frames():

        k = src.current_frame_index
        outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
        frame = src.next_frame()
        
        if not os.path.exists(outfn):
            # compute the B-scan from the spectra, using the provided dispersion coefficients:
            bscan = blobf.spectra_to_bscan(coefs,frame,diagnostics=diagnostics)

            # save the complex B-scan in the B-scan folder
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)
        else:
            logging.info('%s exists. Skipping.'%outfn)

    # Skip this for now. Needs discussion.
    blobf.flatten_volume(bscan_folder,diagnostics=diagnostics)

    flattened_folder = os.path.join(bscan_folder,'flattened')
    
    # Process the ORG blocks
    org_tools.process_org_blocks(flattened_folder,redo=True)
        

if __name__=='__main__':
    unp_file_list = glob.glob('*.unp')
    unp_file_list.sort()


    files_to_process = unp_file_list[:1]
    
    try:
        import multiprocessing as mp
        do_mp = True
        n_cores_available = mp.cpu_count()
        n_cores = n_cores_available-2
        logging.info('n_cores_available: %d'%n_cores_available)
        logging.info('n_cores to be used: %d'%n_cores)
        pool = mp.Pool(n_cores)
        pool.map(process_dataset,files_to_process)
        
    except ImportError as ie:
        for f in files_to_process:
            process_dataset(f)
