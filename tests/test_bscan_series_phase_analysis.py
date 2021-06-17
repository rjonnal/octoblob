import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob
import glob
import os,sys

# setup parameters
folder = './bscan_series/'
filter_string = 'complex*.npy'
dt = 5e-3 # interval between bscans

# set the axial range for phase analysis; if these are None, step 1 below
# will show an average bscan so you can set values for them
z1 = 350
z2 = 400


# get the file list and sort it
filename_list = glob.glob(os.path.join(folder,filter_string))
filename_list.sort()


# get some parameters from the bscans, and set up a time array
temp = np.load(filename_list[0])
sz,sx = temp.shape
N = len(filename_list)
t_arr = np.arange(N)*dt

# step 1 average the unregistered stack to identify a region in which to perform the phase analysis
# use the final figure to set z1 and z2 for the next step
if z1 is None or z2 is None:
    total_mean = np.zeros((sz,sx))
    for f in filename_list:
        bscan = np.load(f)
        total_mean = total_mean + np.abs(bscan)
        plt.cla()
        plt.imshow(np.log10(total_mean),cmap='gray',aspect='auto')
        plt.title(f)
        plt.pause(.001)

    plt.show()

# step 2: track the phase variance as a function of time, without any registration or bulk motion correction
# a reference frame
ref = np.load(filename_list[0])[z1:z2,:]
pv_arr = []
for idx,f in enumerate(filename_list):
    roi = np.load(f)[z1:z2,:]
    pv = np.std(np.angle(roi-ref))**2
    pv_arr.append(pv)

plt.figure()
plt.plot(t_arr,pv_arr,'ks')
plt.xlabel('time (s)')    
plt.ylabel('phase variance')
plt.title('without bulk motion correction')


# step 3: track the phase variance as a function of time, with bulk motion correction
ref = np.load(filename_list[0])[z1:z2,:]
pv_arr = []

stack = np.zeros((z2-z1,sx,2),dtype=np.complex)
stack[:,:,0] = ref

phase_stack = np.zeros((z2-z1,sx,2))
phase_stack[:,:,0] = np.angle(ref)

for idx,f in enumerate(filename_list):
    roi = np.load(f)[z1:z2,:]
    stack[:,:,1] = roi
    stack_mean = np.mean(np.abs(stack),axis=2)

    phase_stack[:,:,1] = np.angle(roi)
    
    threshold = np.mean(stack_mean)+np.std(stack_mean) # selected arbitrarily!

    mask = np.zeros(roi.shape)
    mask[np.where(stack_mean>threshold)] = 1
    
    corrected_phase = blob.bulk_motion_correct(phase_stack,mask,n_bins=8,resample_factor=4,n_smooth=1,diagnostics=False) # check these parameters too!

    # correct the phase of the current roi:
    amplitude = np.abs(roi)
    corrected_complex = amplitude*np.exp(1j*corrected_phase[:,:,1])

    pv = np.std(np.angle(corrected_complex-roi))**2
    pv_arr.append(pv)
    print(idx)
    
plt.figure()
plt.plot(t_arr,pv_arr,'ks')
plt.xlabel('time (s)')    
plt.ylabel('phase variance')
plt.title('with bulk motion correction')
plt.show()
