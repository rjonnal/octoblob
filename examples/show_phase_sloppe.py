import numpy as np
import sys,glob,os
from matplotlib import pyplot as plt

folder = sys.argv[1]

flist = glob.glob(os.path.join(folder,'*phase_slope.npy'))
flist.sort()

imax = -np.inf
imin = np.inf

for f in flist:
    im = np.load(f)
    imax = max(np.nanmax(im),imax)
    imin = min(np.nanmin(im),imin)


for f in flist:
    im = np.load(f)
    plt.clf()
    plt.imshow(im,clim=(imin,imax),cmap='gray')
    plt.ylim((200,150))
    plt.colorbar()
    plt.title(f)
    plt.pause(.1)
