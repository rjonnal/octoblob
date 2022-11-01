import numpy as np
import octoblob.functions as blobf
import octoblob.diagnostics_tools as blobd
from matplotlib import pyplot as plt
import os,sys

make_data = False
folder_name = 'examples/test_2_bscans'

d = blobd.Diagnostics(folder_name)

if make_data:
    fake_data_folder = 'fake_bscans'
    os.makedirs(fake_data_folder,exist_ok=True)

    z = np.arange(100)

    a = []
    for k in range(10):
        a.append(np.sin(z*0.5*np.random.randn()))

    a = np.sum(a,axis=0)

    a[np.where(a>=0)] = 1
    a[np.where(a<0)] = 0

    b = np.array([a]*250).T


    motion = []
    for k in range(10):
        motion.append(np.sin(z*0.1*np.random.randn()))
    motion = np.sum(motion,axis=0)
    motion = np.round(motion).astype(int)

    for idx,m in enumerate(motion):
        print(m)
        temp = np.roll(b,m,axis=0)
        temp = temp + np.random.randn(temp.shape[0],temp.shape[1])
        np.save(os.path.join(fake_data_folder,'bscan_%05d.npy'%idx),temp)

    plt.figure()
    plt.plot(motion)



blobf.flatten_volume(folder_name,diagnostics=d)
