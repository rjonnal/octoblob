from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics_tools
from octoblob import parameters
import sys,os
import numpy as np
from octoblob import mapping_dispersion_optimizer as mdo
from octoblob import file_manager
try:
    import multiprocessing as mp
    do_mp = True
except:
    do_mp = False
    
data_filename = 'test_1.unp'

diagnostics = diagnostics_tools.Diagnostics(data_filename)
#diagnostics = None


# diagnostics gets messed up by parallelism since multiple processes
# may be trying to write to the same file
if diagnostics is not None:
    do_mp = False
    
params_filename = file_manager.get_params_filename(data_filename)

params = parameters.Parameters(params_filename,verbose=True)

if __name__=='__main__':
    src = blobf.get_source(data_filename)

    # try to read dispersion/mapping coefs from a local processing_parameters file, and run optimization otherwise
    try:
        coefs = np.array(params['mapping_dispersion_coefficients'],dtype=np.float)
    except KeyError:
        samples = src.get_samples(5)
        coefs = mdo.multi_optimize(samples,blobf.spectra_to_bscan,show_all=False,show_final=True,verbose=False)
        params['mapping_dispersion_coefficients'] = coefs
    
    # process the bscans
    bscan_folder = file_manager.get_bscan_folder(data_filename)
    if not do_mp:
        for k in range(src.n_total_frames):
            bscan = blobf.spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)
    else:

        def mapping_function(k):
            bscan = blobf.spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)
            outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
            np.save(outfn,bscan)
            logging.info('Saving bscan %s.'%outfn)

        p = mp.Pool(12)
        p.map(mapping_function,list(range(src.n_total_frames)))

    
