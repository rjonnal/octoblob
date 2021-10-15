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


folder_list = ['171358/17_13_58-_bscans/aligned', '171430/17_14_30-_bscans/aligned', '171502/17_15_02-_bscans/aligned', '171544/17_15_44-_bscans/aligned']

args = sys.argv[1:]
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

for folder in folder_list:
    print(folder)
    volume = Volume(folder)
    bscan = np.abs(volume.get_volume()).mean(axis=0)
    prof = np.mean(bscan,axis=1)
    #prof = np.abs(volume.get_volume()).mean(axis=2).mean(axis=0)
    
    profs.append(prof)
    bscans.append(bscan)
    dB_prof = prof/np.max(prof)
    dB_prof = 20*np.log10(dB_prof)
    dB_profs.append(dB_prof)

    dB_bscan = 20*np.log10(bscan)
    dB_bscans.append(dB_bscan)
    
dB_ref = dB_profs[0]

bright_idx = np.where(dB_ref>-15)[0]
rz1 = bright_idx[0]-60
rz2 = bright_idx[-1]+40

ref = profs[0]

for idx,(folder,tar,bscan,dB_bscan) in enumerate(zip(folder_list,profs,bscans,dB_bscans)):
    minlen = min(len(ref),len(tar))
    ref = ref[:minlen]
    tar = tar[:minlen]
    
    nxc = np.real(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))

    shift = np.argmax(nxc)
    if shift>len(nxc)//2:
        shift = shift-len(nxc)
    print(shift)

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
        plt.figure(1)
        plt.plot(tar[tz1:tz2]+200*idx,label='%d'%idx)
        plt.figure()
        plt.imshow(dB_bscan,clim=(40,90),cmap='gray',aspect='auto')
        plt.axhspan(tz1,tz2,color='g',alpha=0.25)
if not write:
    plt.figure(1)
    plt.title("preview of cropped volume profiles\nrun with 'write' as a parameter to perform crop.")
    plt.show()
