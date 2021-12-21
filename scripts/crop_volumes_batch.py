import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from octoblob.volume_tools import Volume, VolumeSeries, Boundaries
from octoblob.ticktock import tick, tock
import octoblob as blob
import octoblob.plotting_functions as opf
opf.setup_plots(mode='paper',style='seaborn-deep')
color_cycle = opf.get_color_cycle()

# The volumes are aligned and cropped by:
# 1. Computing the axial reflectance profile of each volume
# 2. Identifying the pixels that are above a threshold (in dB)
# 3. If z1 and z2 are the indices of the first and last pixel
#    above threshold, the profile is cropped using:
#    z1+inner_padding:z2+outer padding
# 4. Cross-correlating the cropped profiles to align them
# 5. Applying the resulting shifts to z1+inner_padding and
#    z2+outer_padding in order to crop the volumes
# The threshold, inner_padding, and outer_padding can be set here:

threshold_dB = -20
inner_padding = -30
outer_padding = 60
approximate_stimulus_index = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)



args = sys.argv[1:]
args = blob.expand_wildcard_arguments(args)

arg_folders = []
for arg in args:
    if os.path.exists(arg) and len(glob.glob(os.path.join(arg,'complex_bscan*.npy')))>=2:
        arg_folders.append(arg)

if len(arg_folders)>=1:
    folder_list = arg_folders
        
write = any([p.lower()=='write' for p in args])

profs = []
dB_profs = []
bscans = []
dB_bscans = []

uncropped_bscan_fig = plt.figure()
for idx,folder in enumerate(folder_list):
    print(folder)
    volume = Volume(folder)

    try:
        subvolume = volume.get_volume()[approximate_stimulus_index-10:approximate_stimulus_index+10,:,:]
    except Exception as e:
        print(e)
        subvolume = volume.get_volume()
        
    bscan = np.abs(subvolume).mean(axis=0)
    sz,sx = bscan.shape

    x_stop = sx//2
    
    prof = np.mean(bscan[:,:x_stop],axis=1)
    #prof = np.abs(volume.get_volume()).mean(axis=2).mean(axis=0)
    
    profs.append(prof)
    bscans.append(bscan)
    dB_prof = prof/np.max(prof)
    dB_prof = 20*np.log10(dB_prof)
    dB_profs.append(dB_prof)

    dB_bscan = 20*np.log10(bscan)
    dB_bscans.append(dB_bscan)
    plt.plot(dB_prof,label='%d'%idx)

plt.legend()
plt.title('uncropped, unshifted profiles')
plt.axhline(threshold_dB,linestyle='--',color='k')
opf.despine()

dB_ref = dB_profs[0]

bright_idx = np.where(dB_ref>threshold_dB)[0]

rz1 = bright_idx[0]+inner_padding
rz2 = bright_idx[-1]+outer_padding

ref = profs[0]

shifts = []

for idx,(folder,tar,bscan,dB_bscan) in enumerate(zip(folder_list,profs,bscans,dB_bscans)):
    minlen = min(len(ref),len(tar))
    ref = ref[:minlen]
    tar = tar[:minlen]
    
    nxc = np.real(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))

    shift = np.argmax(nxc)
    if shift>len(nxc)//2:
        shift = shift-len(nxc)
    shifts.append(shift)

shifts = np.array(shifts)
    
rz1 = max(0,rz1)
rz2 = min(len(profs[0]),rz2)

shifts = shifts-np.min(shifts)
rz2 = rz2-np.max(shifts)

cropped_bscan_fig = plt.figure()
for idx,(folder,tar,bscan,dB_bscan,shift) in enumerate(zip(folder_list,profs,bscans,dB_bscans,shifts)):
    
    tz1 = rz1+shift
    tz2 = rz2+shift
    if write:
        out_folder = os.path.join(folder,'cropped')
        os.makedirs(out_folder,exist_ok=True)
        flist = sorted(glob.glob(os.path.join(folder,'complex_bscan*.npy')))
        for f in flist:
            basename = os.path.split(f)[1]
            bscan = np.load(f)
            bscan = bscan[tz1:tz2,:]
            out_f = os.path.join(out_folder,basename)
            np.save(out_f,bscan)
            #plt.cla()
            #plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray')
            #plt.pause(0.001)
            print('Cropping %s -> %s.'%(f,out_f))

    else:
        plt.figure(uncropped_bscan_fig.number)
        plt.axvline(tz1,color=color_cycle[idx%len(color_cycle)])
        plt.axvline(tz2,color=color_cycle[idx%len(color_cycle)])
        
        plt.figure(cropped_bscan_fig.number)
        tar = tar/tar.max()
        prof_dB = 20*np.log10(tar[tz1:tz2])

        plt.plot(prof_dB,label='%d'%idx)
        #plt.plot(tar[tz1:tz2],label='%d'%idx)
        plt.figure()
        plt.imshow(dB_bscan,clim=(40,90),cmap='gray',aspect='auto')
        plt.axhspan(tz1,tz2,color='g',alpha=0.15)
        plt.title(idx)
        
if not write:
    plt.figure(cropped_bscan_fig.number)
    plt.axhline(threshold_dB,linestyle='--',color='k')
    plt.legend()
    plt.title("preview of cropped volume profiles\nrun with 'write' as a parameter to perform crop.")
    opf.despine()
    plt.show()
