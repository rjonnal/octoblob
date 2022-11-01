import numpy as np
import octoblob.functions as blobf
import octoblob.diagnostics_tools as blobd
from matplotlib import pyplot as plt
import os,sys

os.makedirs('fake_bscans',exist_ok=True)
d = blobd.Diagnostics('fake_bscans')

z = np.arange(100)

a = []
for k in range(10):
    a.append(np.sin(z*0.3*np.random.randn()))

a = np.sum(a,axis=0)

a[np.where(a>=0)] = 1
a[np.where(a<0)] = 0

b = np.array([a]*250).T


motion = []
for k in range(10):
    motion.append(np.sin(z*0.1*np.random.randn()))
motion = np.sum(motion,axis=0)*10
motion = np.round(motion).astype(int)

for idx,m in enumerate(motion):
    
    temp = np.roll(b,m)
    np.save(os.path.join('fake_bscans','bscan_%05d.npy'%idx),temp)

plt.figure()
plt.plot(motion)



blobf.flatten_volume('fake_bscans',diagnostics=d)


plt.show()
