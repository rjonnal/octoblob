import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from octoblob.volume_tools import Volume, VolumeSeries, gaussian_filter, rect_filter, show3d
from octoblob.ticktock import tick, tock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# how much to resample volumes before registration
resampling = 1.0

# width of gaussian window, in pixels in pre-resampled volume
# i.e., if resampling is 2.0 and sigma is 10, the resulting
# gaussian window would have a width of 20; this is designed to
# avoid having to change sigma when resampling is changed
sigma = 10


def usage():
    print('python average_volumes_2.py reference_folder target_folder_1 target_folder_2 ...')

args = sys.argv[1:]

ref_folder = sys.argv[1]
tar_folders = sys.argv[2:]
print(ref_folder)
vs = VolumeSeries(ref_folder,resampling=resampling,sigma=sigma)
print(ref_folder)

for tar_folder in tar_folders:
    print(tar_folder)
    vs.add_target(tar_folder)

vs.register()

vs.render(threshold_percentile=50.0)
