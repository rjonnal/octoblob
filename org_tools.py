import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys
import octoblob.histogram as blobh
import octoblob.diagnostics_tools as blobd

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

def compute_phase_velocity(stack,diagnostics=None):
    amplitude_mean = np.mean(np.abs(stack),axis=0)
    phase_stack = np.angle(stack)
    mask = blobh.make_mask(amplitude_mean)
    phase_stack = np.transpose(phase_stack,(1,2,0))
    phase_stack = blobh.bulk_motion_correct(phase_stack,mask,diagnostics=diagnostics)

    if diagnostics is not None or True:
        fig = plt.figure()
        plt.subplot(phase_stack.shape[2]+1,1,1)
        plt.imshow(amplitude_mean,aspect='auto',interpolation='none')

        for k in range(phase_stack.shape[2]):
            plt.subplot(phase_stack.shape[2]+1,1,k+2)
            plt.imshow(mask*phase_stack[:,:,k],aspect='auto',interpolation='none')

        diagnostics.save(fig)




if __name__=='__main__':
    
    unp_files = glob.glob('./examples/*.unp')
    for unp_file in unp_files:
        folder = unp_file.replace('.unp','')+'_bscans'
        block_filenames = get_block_filenames(folder)
        for bf in block_filenames:
            d = blobd.Diagnostics(bf[0].replace('.npy',''))
            frames = get_frames(bf)
            compute_phase_velocity(frames,diagnostics=d)
