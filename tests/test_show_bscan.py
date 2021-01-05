import numpy as np
from matplotlib import pyplot as plt
import os,sys

bscan_index = 10
fn = os.path.join('bscan_npy','scan_%03d.npy'%bscan_index)
bscan = np.load(fn)

# simple show:
plt.figure()
plt.imshow(np.abs(bscan),cmap='gray')
plt.colorbar()


# flexible x and y scaling, to fit figure:
plt.figure(figsize=(6,3))
plt.imshow(np.abs(bscan),cmap='gray',aspect='auto')

# contrast limits
upper = np.mean(np.abs(bscan)) + 4*np.std(np.abs(bscan))
lower = np.mean(np.abs(bscan)) + 0.5*np.std(np.abs(bscan))
plt.figure()
plt.imshow(np.abs(bscan),cmap='gray',clim=(lower,upper))

# log, with contrast limits in dB
# reasonable dynamic range is about 40 dB, e.g.
# 40 to 80 dB or 35 to 75 dB
plt.figure(figsize=(4,3))
plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray',clim=(40,80))
plt.colorbar()
plt.savefig('bscan.png',dpi=300)
plt.show()
