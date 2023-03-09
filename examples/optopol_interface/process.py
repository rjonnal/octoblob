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


# For ORG processing we needn't process all the frames. 400 frames are acquired
# in each measurememnt, at a rate of 400 Hz. The stimulus onset is at t=0.25 s,
# corresponding to the 100th frame. 50 milliseconds before stimulus is sufficient
# to establish the baseline, and the main ORG response takes place within 100
# milliseconds of the stimulus. Thus:
org_start_frame = 0
org_end_frame = 100

# Enter this here or pass at command line
amp_filename = None
bscan_height = 320

if amp_filename is None:
    try:
        amp_filename = sys.argv[1]
    except IndexError as ie:
        sys.exit('Please check amp_filename. %s not found or amp_filename not passed at command line.'%amp_filename)

phase_filename = amp_filename.replace('_Amp.bin','_Phase.bin')

org_frames = list(range(org_start_frame,org_end_frame))

# Create a diagnostics object for inspecting intermediate processing steps
diagnostics = diagnostics_tools.Diagnostics(amp_filename)

# Create a parameters object for storing and loading processing parameters
params_filename = file_manager.get_params_filename(amp_filename)
params = parameters.Parameters(params_filename,verbose=True)

# get the folder name for storing bscans
bscan_folder = file_manager.get_bscan_folder(amp_filename)


temp = np.fromfile(amp_filename,dtype=np.uint8,count=12)

dims = np.fromfile(amp_filename,dtype=np.int32,count=3)

n_depth,n_fast,n_slow = dims

assert org_start_frame<n_slow
assert org_end_frame<=n_slow

def get_cube(fn,show_hist=False):
    dat = np.fromfile(fn,dtype=np.dtype('<f4'),offset=12,count=n_depth*n_fast*n_slow)
    if show_hist:
        plt.hist(dat)
        plt.show()
    dat = np.reshape(dat,dims[::-1])
    dat = np.transpose(dat,(0,2,1))
    dat = dat[:,::-1,:]
    return dat.astype(np.float)

amp = get_cube(amp_filename)
phase = get_cube(phase_filename)

height = 320
bmean = np.mean(amp,axis=0)
z1,z2 = blobf.get_bscan_boundaries(bmean,height)

for k in range(org_start_frame,org_end_frame):
    bamp = blobf.insert_bscan(amp[k,:,:],z1,z2,height)
    bphase = blobf.insert_bscan(phase[k,:,:],z1,z2,height)
    bscan = bamp*np.exp(bphase*1j)
    bscan = bscan.astype(np.complex64)
    #bscan = amp[k,:,:]
    dB = blobf.dB(bscan)
    bphase = np.angle(bscan)
    plt.clf()
    plt.imshow(dB,clim=(45,90),cmap='gray')
    plt.colorbar()
    plt.pause(.01)
    outfn = os.path.join(bscan_folder,file_manager.bscan_template%k)
    np.save(outfn,bscan)
    
    print(bscan.dtype)
    
# Skip this for now. Needs discussion.
#blobf.flatten_volume(bscan_folder,diagnostics=diagnostics)

# Process the ORG blocks
org_tools.process_org_blocks(bscan_folder)
        
