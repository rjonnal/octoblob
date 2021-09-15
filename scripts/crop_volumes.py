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

write = any([p.lower()=='write' for p in sys.argv[1:]])

profs = []
dB_profs = []

for directory in folder_list:
    volume = Volume(directory)
    prof = np.abs(volume.get_volume()).mean(axis=2).mean(axis=0)
    profs.append(prof)
    dB_prof = prof/np.max(prof)
    dB_prof = 20*np.log10(dB_prof)
    dB_profs.append(dB_prof)
    
dB_ref = dB_profs[0]

bright_idx = np.where(dB_ref>-15)[0]
rz1 = bright_idx[0]-40
rz2 = bright_idx[-1]+40

ref = profs[0]

for idx,(directory,tar) in enumerate(zip(folder_list,profs)):
    minlen = min(len(ref),len(tar))
    ref = ref[:minlen]
    tar = tar[:minlen]
    
    nxc = np.real(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))

    shift = np.argmax(nxc)
    if shift>len(nxc)//2:
        shift = shift+len(nxc)
    print(shift)

    tz1 = rz1+shift
    tz2 = rz2+shift

    if write:
        out_directory = os.path.join(directory,'cropped')
        os.makedirs(out_directory,exist_ok=True)
        flist = sorted(glob.glob(os.path.join(directory,'complex_bscan*.npy')))
        for f in flist:
            basename = os.path.split(f)[1]
            bscan = np.load(f)
            bscan = bscan[tz1:tz2,:]
            out_f = os.path.join(out_directory,basename)
            np.save(out_f,bscan)
            plt.cla()
            plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray')
            plt.pause(0.001)
            print('Cropping %s -> %s.'%(f,out_f))

    else:
        plt.plot(tar[tz1:tz2]+200*idx,label='%d'%idx)

if not write:        
    plt.title("preview of cropped volume profiles\nrun with 'write' as a parameter to perform crop.")
    plt.show()
