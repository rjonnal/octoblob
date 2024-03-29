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

data_filename = None

if data_filename is None:
    try:
        data_filename = sys.argv[1]
    except IndexError as ie:
        sys.exit('Please check data_filename. %s not found or data_filename not passed at command line.'%data_filename)


# For ORG processing we needn't process all the frames. 400 frames are acquired
# in each measurememnt, at a rate of 400 Hz. The stimulus onset is at t=0.25 s,
# corresponding to the 100th frame. 50 milliseconds before stimulus is sufficient
# to establish the baseline, and the main ORG response takes place within 100
# milliseconds of the stimulus. Thus:
org_start_frame = 0
org_end_frame = 140

org_frames = list(range(org_start_frame,org_end_frame))

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

for k in range(src.n_total_frames):
    # skip this frame if it's not in the ORG frame range
    if not k in org_frames:
        continue
    # compute the B-scan from the spectra, using the provided dispersion coefficients:
    outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
    if os.path.exists(outfn):
        continue
    bscan = blobf.spectra_to_bscan(coefs,src.get_frame(k),diagnostics=diagnostics)

    # save the complex B-scan in the B-scan folder
    np.save(outfn,bscan)
    logging.info('Saving bscan %s.'%outfn)

# Skip this for now. Needs discussion.
#blobf.flatten_volume(bscan_folder,diagnostics=diagnostics)

# Process the ORG blocks
org_tools.process_org_blocks(bscan_folder)
        
