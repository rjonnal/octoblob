import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1


def savebmp(filename,arr,clim=None,dpi=100,cmap='gray'):
    # only work for 2D arrays:
    try:
        assert len(arr.shape)==2
    except AssertionError as ae:
        print('savebmp requires 2D array only but was given an array with shape',arr.shape)

    if clim is None:
        clim = (np.min(arr),np.max(arr))

    arr = (arr-clim[0])/(clim[1]-clim[0])
    arr = np.clip(arr,0.0,1.0)
    arr = arr*255.0
    arr = np.uint8(np.round(arr))

    sy,sx = arr.shape

    syinches,sxinches = float(sy)/float(dpi),float(sx)/float(dpi)
    
    plt.figure(figsize=(sxinches,syinches))
    plt.axes([0,0,1,1])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(arr,cmap=cmap)
    plt.savefig(filename,dpi=dpi)
    plt.close()

def dbscale(frame):
    '''A convenience function for scaling OCT b-scans
    expects a complex, linear-scale matrix. Returns dB'''
    return 20*np.log10(np.abs(frame))
    
def logscale(frame,lower_limit=None,upper_limit=None,bit_depth=16):
    '''A convenience function for scaling OCT b-scans
    expects a complex, linear-scale matrix. Returns a
    rounded value scaled to the desired bit depth.'''

    frame = np.log(np.abs(frame))
    if lower_limit is None:
        lower_limit = np.median(frame)-1.0*np.std(frame)
    if upper_limit is None:
        upper_limit = np.median(frame)+4*np.std(frame)
    
    return np.round(((frame - lower_limit)/(upper_limit-lower_limit)*2**bit_depth)).clip(0,2**bit_depth-1)

def linearscale(frame,lower_limit=None,upper_limit=None,bit_depth=16):
    '''A convenience function for scaling OCT b-scans
    expects a complex, linear-scale matrix. Returns a
    rounded value scaled to the desired bit depth.'''
    frame = np.abs(frame)
    if lower_limit is None:
        lower_limit = np.median(frame)+0.5*np.std(frame)
    if upper_limit is None:
        upper_limit = np.median(frame)+3.5*np.std(frame)
    
    return np.round(((frame - lower_limit)/(upper_limit-lower_limit)*2**bit_depth)).clip(0,2**bit_depth-1)

def add_colorbar(im, aspect=10, pad_fraction=0.1, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
