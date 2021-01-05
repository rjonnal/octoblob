from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import octoblob as blob

# PARAMETERS FOR RAW DATA SOURCE
filename = './octa_test_set.unp'
n_vol = 1



# Here, let's use the actual number of B-scans in the file
n_slow = 20
n_repeats = 1

# Specify the noise region in x and z
noise_x1 = 20
noise_x2 = 380
noise_z1 = 100
noise_z2 = 200

# Which B-scans to use? Valid values are 'odd', 'even', and 'both'
bscan_sequence = 'both'

n_fast = 500
n_skip = 500
n_depth = 1536
bit_shift_right = 4
dtype=np.uint16

fbg_position = 148
spectrum_start = 159
spectrum_end = 1459

L0 = 1060e-9
n = 1.38

src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=fbg_position,spectrum_start=spectrum_start,spectrum_end=spectrum_end,bit_shift_right=bit_shift_right,n_skip=n_skip,dtype=dtype)

# PROCESSING PARAMETERS
mapping_coefficients = [12.5e-10,-12.5e-7,0.0,0.0]
dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]

fft_oversampling_size = 4096
bscan_z1 = 3147
bscan_z2 = -40
bscan_x1 = 0
bscan_x2 = -100

# In this section, we will load one set of repeats and arrange them in a 3D array
# to be bulk-motion corrected

all_phase_jumps = []
all_theoretical_phase_sensitivities = []

try:
    os.mkdir('bscan_npy')
except Exception as e:
    pass

if bscan_sequence.lower()=='both':
    start = 0
    stride = 1
elif bscan_sequence.lower()=='even':
    start = 0
    stride = 2
elif bscan_sequence.lower()=='odd':
    start = 1
    stride = 2

for frame_index in range(start,n_slow-stride,stride):
    print('frame %d of %d'%(frame_index,n_slow-stride))
    npy_filename = os.path.join('bscan_npy','scan_%03d.npy'%frame_index)
    frame1 = src.get_frame(frame_index)
    frame2 = src.get_frame(frame_index+stride)
    bscans = []
    for frame in [frame1,frame2]:
        frame = blob.dc_subtract(frame)
        frame = blob.k_resample(frame,mapping_coefficients)
        frame = blob.dispersion_compensate(frame,dispersion_coefficients)
        frame = blob.gaussian_window(frame,0.9)
        bscan = blob.spectra_to_bscan(frame,oversampled_size=fft_oversampling_size,z1=bscan_z1,z2=bscan_z2)
        bscans.append(bscan)

    sz,sx = bscan.shape
    # compute SNR
    noise_region = np.abs(bscan[noise_z1:noise_z2,noise_x1:noise_x2])
    snr1 = np.max(np.abs(bscan),axis=0)/np.mean(noise_region)
    snr2 = np.max(np.abs(bscan),axis=0)/np.std(noise_region)
    theoretical_phase_sensitivity = np.sqrt(1/snr2)
    theoretical_displacement_sensitivity = L0/(4*n*np.pi)*theoretical_phase_sensitivity
    
    all_theoretical_phase_sensitivities.append(theoretical_phase_sensitivity.T)
    
    if frame_index==0:
        plt.imshow(np.log(np.abs(bscans[0])),cmap='gray')
        plt.autoscale(False)
        plt.axvspan(noise_x1,noise_x2,ymin=1.0-float(noise_z1)/float(sz),ymax=1.0-float(noise_z2)/float(sz),alpha=0.25)
        #plt.plot([noise_x1,noise_x2,noise_x2,noise_x1,noise_x1],
        #         [noise_z1,noise_z1,noise_z2,noise_z2,noise_z1],
        #         'r-')
        plt.title('noise region')
        plt.figure()
        plt.plot(snr1,label='mean noise')
        plt.plot(snr2,label='std noise')
        plt.legend()
        plt.show()
        
    np.save(npy_filename,bscan)
    bscans = np.array(bscans)
    
    stack_complex = np.transpose(bscans,(1,2,0))

    stack_amplitude = np.abs(stack_complex)
    stack_log_amplitude = 20*np.log10(stack_amplitude)
    stack_phase = np.angle(stack_complex)
    
    CSTD = np.std(np.mean(stack_log_amplitude,2))
    FMID = np.mean(np.mean(stack_log_amplitude,2))
    stack_log_amplitude = stack_log_amplitude-(FMID-0.9*CSTD)
    
    stack_log_amplitude = stack_log_amplitude/stack_log_amplitude.max()
    stack_amplitude = stack_amplitude/stack_amplitude.max()

    stack_log_amplitude[stack_log_amplitude<0] = 0.0

    mean_log_amplitude_stack = np.mean(stack_log_amplitude,2)
    phase_stability_threshold = 0.3
    phase_stability_mask = (mean_log_amplitude_stack>phase_stability_threshold)

    plt.imshow(phase_stability_mask)
    plt.savefig('phase_stability_mask.png')
    
    phase_jumps = blob.get_phase_jumps(stack_phase,phase_stability_mask)
    all_phase_jumps.append(phase_jumps.T)
    #print(np.array(all_phase_jumps).shape)
    
all_phase_jumps = np.squeeze(np.array(all_phase_jumps))
all_theoretical_phase_sensitivities = np.squeeze(np.array(all_theoretical_phase_sensitivities))

np.save('theoretical_phase_sensitivity.npy',all_theoretical_phase_sensitivities)
np.save('measured_phase_instability.npy',all_phase_jumps)


plt.figure()
plt.imshow(all_phase_jumps,aspect='auto')
plt.title('phase error between B-scans (rad)')
plt.xlabel('fast scan direction')
plt.ylabel('B-scan number')
plt.colorbar()
plt.savefig('measured_phase_instability.png')

plt.figure()
plt.imshow(all_theoretical_phase_sensitivities,aspect='auto')
plt.title('theoretical phase sensitivity (rad)')
plt.xlabel('fast scan direction')
plt.ylabel('B-scan number')
plt.colorbar()
plt.savefig('theoretical_phase_sensitivity.png')

# trim edges, which have weird artifacts, and plot average along B-scan
average_phase_jumps = np.mean(all_phase_jumps[:,3:-3],axis=1)
plt.figure()
plt.plot(average_phase_jumps)
plt.xlabel('B-scan number')
plt.ylabel('average phase shift between B-scans (rad)')


plt.show()
