import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob

input_directory = 'angio/14_56_58-_bscans/registered'

#contrast limits:
pct_low = 30.0
pct_high = 99.5

microns_per_pixel = 7.0 # not sure of this number

flist = glob.glob(os.path.join(input_directory,'*.npy'))
flist.sort()

vol = []
for f in flist:
    arr = np.load(f)

    # in case data are complex:
    arr = np.abs(arr)
    
    # in case we're working with a stack:
    try:
        assert len(arr.shape)==2
    except AssertionError as ae:
        print('Averaging stack in slow/BM direction.')
        arr = arr.mean(2)
        
    vol.append(arr)

vol = np.array(vol)
global_clim = np.percentile(vol,(pct_low,pct_high))

# function to remove plot spines:
def despine(ax=None):
    """Remove the spines from a plot. (These are the lines drawn
    around the edge of the plot.)"""
    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)



# dimensions of volume are (slow_scan,depth,fast_scan)

# create an axial plot--average first in dimension 2, and then in dimension 0:
axial_profile = vol.mean(2).mean(0) # could also write np.mean(np.mean(vol,2),0)
z = np.arange(len(axial_profile))*microns_per_pixel
plt.figure(figsize=(8,6),dpi=100)
plt.plot(z,axial_profile)

# create an axial plot over a small portion of the volume to better resolve layers:
axial_profile = vol[:50,:,:50].mean(2).mean(0)
plt.figure(figsize=(8,6),dpi=100)
plt.plot(z,axial_profile)

# examples of annotation; may or may not be useful; remember coordinates given in units of the plotted data:
plt.axvline(1000,color='r')
plt.axvspan(500,750,color='g',alpha=0.1)
plt.text(300,300,'foobar')

plt.xlabel('z ($\mu m$)') # use $$ to enclose latex
plt.ylabel('amplitude (ADU)')
despine()


# do some b-scan projections by averaging 100 b-scans together:

# strict aspect ratio
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(vol[:100,:,:].mean(0),cmap='gray')

# if you want to relax the aspect ratio:
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(vol[:100,:,:].mean(0),cmap='gray',aspect='auto')

# over smaller number of B-scans, say 10:
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(vol[:10,:,:].mean(0),cmap='gray',aspect='auto')

# or average slow scans together; need the .T to transpose the result
# otherwise the imshow y dimension is the fast scan
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(vol[:,:,20:30].mean(2).T,cmap='gray',aspect='auto')

# define a b-scan and look at log scale
bscan = vol[20:30,:,:].mean(0)
db = 20*np.log10(bscan)
db_limits = (40,80)
plt.figure(figsize=(8,6),dpi=100)
plt.imshow(db,clim=db_limits,cmap='gray',aspect='auto')
plt.colorbar()




plt.show()


