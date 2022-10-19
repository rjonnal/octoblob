from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics_tools
from octoblob import parameters
import sys,os,glob
import numpy as np
from octoblob import mapping_dispersion_optimizer as mdo
from octoblob import file_manager
import pathlib
    
try:
    import multiprocessing as mp
    do_mp = True
    n_cores_available = mp.cpu_count()
    n_cores = n_cores_available-2
    logging.info('n_cores_available: %d'%n_cores_available)
    logging.info('n_cores to be used: %d'%n_cores)
except:
    do_mp = False

try:
    with open('crawler_blacklist','r') as fid:
        crawler_blacklist = [f.strip() for f in fid.readlines()]
        logging.info('crawler_blacklist found: %s'%crawler_blacklist)
except FileNotFoundError as fnfe:
    crawler_blacklist = []
    logging.info('no crawler_blacklist found')
    
org_frames_only = True
org_frames = list(range(20,80))
do_all_frames_tag = 'fovea'

start_clean = 'clean' in sys.argv[1:]


def process(data_filename,do_mp=False):

    data_filename = str(data_filename)
    diagnostics = diagnostics_tools.Diagnostics(data_filename)

    # diagnostics gets messed up by parallelism since multiple processes
    # may be trying to write to the same file
    if diagnostics is not None:
        do_mp = False

    params_filename = file_manager.get_params_filename(data_filename)
    params = parameters.Parameters(params_filename,verbose=True)

    src = blobf.get_source(data_filename)

    # get the total number of frames:
    if org_frames_only:
        n_total_frames = len(org_frames)
    else:
        n_total_frames = src.n_total_frames

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
    # check to see how many bscans there are in it:
    bscans = glob.glob(os.path.join(bscan_folder,'*.npy'))
    # if any are missing, reprocess:
    if len(bscans)<n_total_frames:
        logging.info('File %s missing B-scans. Re-processing.'%data_filename)
        for k in range(src.n_total_frames):
            if org_frames_only and not k in org_frames: and data_filename.lower().find(do_all_frames_tag)==-1:
                continue
            bscan = blobf.spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)
    else:
        logging.info('File %s B-scans processed. Skipping.'%data_filename)



if __name__=='__main__':

    if start_clean:
        file_manager.clean(False)
        file_manager.clean(True)
        
    unp_files_temp = pathlib.Path('.').rglob('*.unp')
    unp_files_temp = [str(f) for f in unp_files_temp]
    unp_files = []
    for unp_file in unp_files_temp:
        file_blacklisted = False
        for item in crawler_blacklist:
            if unp_file[:len(item)]==item:
                logging.info('blacklisted %s for matching %s'%(unp_file,item))
                file_blacklisted = True
        if not file_blacklisted:
            unp_files.append(unp_file)

    logging.info('Processing these files:')
    for uf in unp_files:
        logging.info('\t %s'%uf)

    def multiprocessing_function(f):
        logging.info('Crawling %s.'%f)
        try:
            process(f)
        except Exception as e:
            logging.info('Error: %s. Skipping %s.'%(e,f))
            

    if do_mp:
        p = mp.Pool(n_cores)
        p.map(multiprocessing_function,unp_files)

    else:
    
        for unp_file in unp_files:
            logging.info('Crawling %s.'%unp_file)
            try:
                process(unp_file)
            except Exception as e:
                logging.info(e)
