import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys

# some parameters for limiting processing of B-scans
org_stimulus_frame = 100
org_start_frame = 80
org_end_frame = 140


def get_block_filenames(folder,file_filter='*.npy',block_size=5):
    files = sorted(glob.glob(os.path.join(folder,file_filter)))
    first_first = 0
    last_first = len(files)-block_size
    out = []
    for k in range(first_first,last_first):
        out.append(list(files[k] for k in list(range(k,k+block_size))))
    return out

def get_frames(filename_list):
    stack = []
    for f in filename_list:
        stack.append(np.load(f))
    stack = np.array(stack)
    return stack

def compute_phase_velocity(stack):
    amplitude_mean = np.mean(np.abs(stack),axis=0)

    

    plt.imshow(amplitude_mean)
    plt.show()
    sys.exit()

