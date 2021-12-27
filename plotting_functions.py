import numpy as np
from matplotlib import pyplot as plt
import sys,os
from matplotlib import animation

print_dpi = 300
screen_dpi = 100
IPSP = 2.0

def mdsavefig(fn,dpi=print_dpi):
    plt.savefig(fn,dpi=dpi)
    print('![](%s)'%fn)

def pad(ax=None,frac=0.1):
    """Add some vertical padding to a plot."""
    if ax is None:
        ax = plt.gca()
    ymin = np.inf
    ymax = -np.inf
    for line in ax.lines:
        yd = line.get_ydata()
        if yd.min()<ymin:
            ymin = yd.min()
        if yd.max()>ymax:
            ymax = yd.max()
            
    yr = ymax-ymin
    ylim = ((ymin-yr*frac,ymax+yr*frac))
    ax.set_ylim(ylim)
    

def dots(ax=None,border_fraction=0.03,markersize=4):
    if ax is None:
        ax = plt.gca()
        
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    lines = ax.lines

    xmax = -np.inf
    ymax = -np.inf
    xmin = np.inf
    ymin = np.inf

    ymin,ymax = ylim
    xmin,xmax = xlim
    
    doty = (ymin+ymax)/2.0
    xr = xmax-xmin
    leftx = np.linspace(xmin+xr*border_fraction*0.1,xmin+xr*border_fraction,3)
    rightx = np.linspace(xmax-xr*border_fraction*0.1,xmax-xr*border_fraction,3)
    for lx in leftx:
        ax.plot(lx,doty,'k.',markersize=markersize)
        print(lx,doty)
    for rx in rightx:
        ax.plot(rx,doty,'k.',markersize=markersize)
    
    
def despine(ax=None):
    """Remove the spines from a plot. (These are the lines drawn
    around the edge of the plot.)"""
    if ax is None:
        ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

def setup_plots(mode='paper',style='seaborn-deep',font_size=9,font='Arial'):
    
    if font=='Times New Roman':
        plt.rc('font', family='serif')
        plt.rc('font', serif=['Times New Roman'])
    elif font=='Arial':
        plt.rc('font', family='sans-serif')
        plt.rcParams['font.sans-serif'] = 'Arial'

    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)  # fontsize of the figure title
    plt.style.use(style)

def get_color_cycle():
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return color_cycle
