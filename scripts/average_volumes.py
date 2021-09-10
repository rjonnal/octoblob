import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from octoblob.volume_tools import Volume, VolumeSeries, Boundaries
from octoblob.ticktock import tick, tock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


M_SCAN_DIMENSION = 2
t0 = None

#reference_bscan_directory = '171358/17_13_58-_bscans/aligned/'
#target_bscan_directories = ['171430/17_14_30-_bscans/aligned','171502/17_15_02-_bscans/aligned','171544/17_15_44-_bscans/aligned']


#all_directories = sorted(glob.glob('17*/17*_bscans/aligned'))
#all_directories = sorted(glob.glob(os.path.join('synthetic','*')))
all_directories = sorted(glob.glob(os.path.join('synthetic_rigid','*')))
reference_bscan_directory = all_directories[0]
target_bscan_directories = all_directories[1:]


reference_volume = Volume(reference_bscan_directory,diagnostics=False)
vseries = VolumeSeries()
vseries.add(reference_volume)
downsample = 5

for target_bscan_directory in target_bscan_directories:
    target_volume = Volume(target_bscan_directory)
    target_volume.register_to(reference_volume,downsample=5)
    vseries.add(target_volume)

# step through the volume in chunks of 10 B-scans
for v in vseries[1:]:
    block_size = 30
    starts = [s for s in range(0,v.n_slow,block_size)]
    ends = [s+block_size for s in starts]
    ends[-1] = min(ends[-1],v.n_slow) # the last block may not have block_size B-scans in it

    for s,e in zip(starts,ends):
        b = Boundaries(s,e,0,v.n_depth,0,v.n_fast)
        reference_volume.get_block(b)
        v.get_block(b)
        plt.show()
        

    
    sys.exit()


vseries.render(True)
sys.exit()


plt.figure()
plt.subplot(1,3,1)
plt.imshow(nxc[reg_coords[0],:,:])
plt.subplot(1,3,2)
plt.imshow(nxc[:,reg_coords[1],:])
plt.subplot(1,3,3)
plt.imshow(nxc[:,:,reg_coords[2]])
plt.show()






