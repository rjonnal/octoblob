import numpy as np
from matplotlib import pyplot as plt
import sys,os

def smooth3(dat,rt=13,rx=13,rz=0):
    st,sz,sx = dat.shape
    tt,zz,xx = np.meshgrid(np.arange(st),np.arange(sz),np.arange(sx),indexing='ij')
    tt = tt - st/2.0
    zz = zz - sz/2.0
    xx = xx - sx/2.0

    k = np.zeros(dat.shape)
    valid = (np.abs(tt)<=rt) * (np.abs(zz)<=rz) * (np.abs(xx)<=rx)
    k[valid] = 1.0

    out = np.fft.ifftn(np.fft.fftn(dat)*np.fft.fftn(k))
    out = np.fft.fftshift(out)
    
    out = out/np.sum(k)
    
    if all(np.isreal(dat.ravel())):
        out = np.real(out)

    return out

def smooth2(dat,rx=5,rz=0):
    sz,sx = dat.shape
    zz,xx = np.meshgrid(np.arange(sz),np.arange(sx),indexing='ij')
    zz = zz - sz/2.0
    xx = xx - sx/2.0

    k = np.zeros(dat.shape)
    valid = (np.abs(zz)<=rz) * (np.abs(xx)<=rx)
    k[valid] = 1.0

    out = np.fft.ifft2(np.fft.fft2(dat)*np.fft.fft2(k))
    out = np.fft.fftshift(out)
    
    out = out/np.sum(k)
    
    if all(np.isreal(dat.ravel())):
        out = np.real(out)

    return out

def dB(a):
    return 20*np.log10(np.abs(a))

def get_peaks(bscan,fractional_threshold=0.5,width=25,region=1,gradient_fractional_threshold=0.0):
    # identify tall peaks, above threshold and highest within +/- region
    prof = np.mean(bscan[:,:width],axis=1)
    left = prof[:-2]
    center = prof[1:-1]
    right = prof[2:]
    thresh = np.max(prof)*fractional_threshold
    grad_thresh = np.max(prof)*gradient_fractional_threshold
    peak_idx_all = np.where((center>thresh) * (center>left+grad_thresh) * (center>right+grad_thresh))[0]+1
    peak_idx = []
    for p in peak_idx_all:
        temp = np.zeros(prof.shape)
        temp[:] = prof[:]
        temp[:p-region] = 0
        temp[p+region+1:] = 0

        # plt.figure()
        # plt.plot(prof)
        # plt.plot(temp)
        if temp[p]==np.max(temp):
            peak_idx.append(p)
        #     plt.title('yes')
        # else:
        #     plt.title('no')
        # plt.show()
            
    return peak_idx, prof


def find_path(bscan,z_start,x_start=0,show=False,layer_half_width=0,diag_weight=0.75,vert_weight=-np.inf):

    if type(show)==str:
        col = show
    else:
        col = 'r'
    
    
    sz,sx = bscan.shape

    # -np.inf for the pixels above and below
    # forces the search to go rightward one
    # pixel on every step
    horiz_weight = 1.0
    search_matrix_r = [(-1,0,vert_weight),
                       (-1,1,diag_weight),
                       (0,1,horiz_weight),
                       (1,0,vert_weight),
                       (1,1,diag_weight)]
    z = z_start
    x = x_start

    path = {'x':[],'z':[]}

    while True:
        path['x'].append(x)
        path['z'].append(z)
        scores = []
        for s in search_matrix_r:
            scores.append(bscan[z+s[0],x+s[1]]*s[2])

        s = search_matrix_r[np.nanargmax(scores)]

        bscan[z-layer_half_width:z+layer_half_width+1,x] = np.nan

        z = z + s[0]
        x = x + s[1]

        if show and (x%10==0 or (sx-x)<2):
            plt.cla()
            plt.imshow(bscan,cmap='gray')
            plt.plot(path['x'],path['z'],'%s-'%col,alpha=0.75,linewidth=2)
            plt.pause(.1)
        
        if x==sx-1:
            break

    path['x'] = np.array(path['x']).astype(int)
    path['z'] = np.array(path['z']).astype(int)

    return path


def trace(bscan,fractional_threshold=0.5,layer_half_width=0):

    sz,sx = bscan.shape

    peak_idx,prof = get_peaks(bscan,fractional_threshold)

    if True: #False and not len(peak_idx)==3:
        plt.figure()
        plt.plot(prof)
        plt.plot(peak_idx,prof[peak_idx],'ro')
        plt.show()
    
    # -np.inf for the pixels above and below
    # forces the search to go rightward one
    # pixel on every step
    vert_weight = -np.inf
    diag_weight = 0.5
    horiz_weight = 1.0
    search_matrix_r = [(-1,0,vert_weight),
                       (-1,1,diag_weight),
                       (0,1,horiz_weight),
                       (1,0,vert_weight),
                       (1,1,diag_weight)]
    paths = {}

    for z0 in peak_idx:
        z = z0
        x = 0

        paths[z0] = {'x':[],'z':[]}

        while True:
            paths[z0]['x'].append(x)
            paths[z0]['z'].append(z)
            scores = []
            for s in search_matrix_r:
                scores.append(bscan[z+s[0],x+s[1]]*s[2])

            s = search_matrix_r[np.nanargmax(scores)]

            bscan[z-layer_half_width:z+layer_half_width+1,x] = np.nan
            
            z = z + s[0]
            x = x + s[1]

            if x==sx-1:
                break

    return paths


if __name__=='__main__':
    vol = np.load('volume_2.0.npy')
    avol = np.abs(vol)

    for t1 in range(5,395,5):
        t2 = t1 + 5
        bscan = np.mean(avol[t1:t2,:,:],axis=0)
        bscan = smooth2(bscan)
        paths = trace(bscan)
        
        plt.cla()
        plt.imshow(dB(bscan),cmap='gray')
        for peak in paths.keys():
            x,z = paths[peak]['x'],paths[peak]['z']
            plt.plot(x,z,alpha=0.5)
        plt.pause(.1)
