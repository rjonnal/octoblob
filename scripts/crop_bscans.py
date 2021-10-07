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

args = sys.argv[1:]

def usage():
    print('Usage: python crop_bscans.py bscan_folder left_edge right_edge')
    print('E.g., python crop_bscans.py 11_22_33_bscans 20 200')
    print('The previous usage will crop the leftmost 20 columns and any columns')
    print('  greater or equal to 200.')
    sys.exit()

if len(args)<3:
    usage()

arg_folders = []
arg_numbers = []
write = False
for arg in args:
    if arg.lower()=='write':
        write = True
    if os.path.exists(arg) and len(glob.glob(os.path.join(arg,'complex_bscan*.npy')))>=2:
        arg_folders.append(arg)
    else:
        try:
            arg_numbers.append(int(arg))
        except:
            pass

try:
    left,right = arg_numbers
except:
    usage()

def show_crop(b):
    plt.subplot(2,1,1)
    plt.imshow(b)
    b = b[:,left:right]
    plt.subplot(2,1,2)
    plt.imshow(b)
    plt.show()

for folder in arg_folders:
    flist = glob.glob(os.path.join(folder,'*bscan*.npy'))

    for f in flist:
        b = np.load(f)
        ndim = len(b.shape)
        if not write:
            if ndim==2:
                show_crop(20*np.log10(np.abs(b)))
            elif ndim==3:
                show_crop(20*np.log10(np.abs(b).mean(2)))
            sys.exit()
        else:
            if ndim==2:
                b = b[:,left:right]
            elif ndim==3:
                b = b[:,left:right,:]
            print('Saving cropped to %s.'%f)
            np.save(f,b)
