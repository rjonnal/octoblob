import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from octoblob.volume_tools import Volume, VolumeSeries, Boundaries, gaussian_filter, rect_filter, show3d
from octoblob.ticktock import tick, tock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
t0 = None


folder_list = ['171358/17_13_58-_bscans/aligned/cropped', '171430/17_14_30-_bscans/aligned/cropped', '171502/17_15_02-_bscans/aligned/cropped', '171544/17_15_44-_bscans/aligned/cropped']

#folder_list = ['synthetic/synthetic_000', 'synthetic/synthetic_001', 'synthetic/synthetic_002', 'synthetic/synthetic_003', 'synthetic/synthetic_004']

# which item in the folder_list is the reference volume?
reference_index = 0

# rendering functions (for PNG B-scans only)
display_function = lambda x: 20*np.log10(np.abs(x)) # convert to dB
# display_function = lambda x: np.abs(x)

# contrast limits
display_clim = (40,80) # dB
# display_clim = None


reference_bscan_directory = folder_list[reference_index]
reference_volume = Volume(reference_bscan_directory,diagnostics=False)
tag = 'reference_%s'%os.path.split(reference_bscan_directory)[1]
n_bscans = reference_volume.n_slow


##############################################################################
block_size_list = [n_bscans,10,2]
block_downsample_list = [5,2,1]
block_filter_sigma_list = [10.0,5.0,2.0]
##############################################################################


vseries = VolumeSeries()
for directory in folder_list:
    vseries.add(Volume(directory,diagnostics = False))

# render the unregistered series, for comparison:
vseries.render('%s_unregistered'%tag,True)

for block_size,block_downsample,block_filter_sigma in zip(block_size_list,block_downsample_list,block_filter_sigma_list):
    # step through the volume in chunks

    #filt = rect_filter((block_size//block_downsample,reference_volume.n_depth//block_downsample,reference_volume.n_fast//block_downsample),(block_filter_sigma,block_filter_sigma,block_filter_sigma))
    filt = gaussian_filter((block_size,reference_volume.n_depth,reference_volume.n_fast),(block_filter_sigma,block_filter_sigma,block_filter_sigma))

    filt = filt[::block_downsample,::block_downsample,::block_downsample]
    
    # use all of vseries instead of vseries[1:], as a sanity
    # check; vseries[0] should result in all 0s, for all block sizes
    for v in vseries:
        starts = [s for s in range(0,v.n_slow,block_size)]
        ends = [s+block_size for s in starts]
        ends[-1] = min(ends[-1],v.n_slow) # the last block may not have block_size B-scans in it

        for s,e in zip(starts,ends):
            b = Boundaries(y1=s,y2=e,z1=0,z2=v.n_depth,x1=0,x2=v.n_fast)
            v.register_to(reference_volume,b,downsample=block_downsample,diagnostics=False,nxc_filter=filt)
            logging.info('')

    # render at each block size:
    vseries.render('%s_strip_registered_%03d'%(tag,block_size),True)

