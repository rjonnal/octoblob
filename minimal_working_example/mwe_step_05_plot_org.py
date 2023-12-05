import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from config import stimulus_index,bscan_interval,dB_clims

def dB(arr):
    return 20*np.log10(np.abs(arr))


if __name__=='__main__':
    org_data_folder = sys.argv[1]
    org_data = np.loadtxt(os.path.join(org_data_folder,'org_data.txt'))
    amplitude_correlations = np.loadtxt(os.path.join(org_data_folder,'amplitude_correlations.txt'))

    bscan = np.load(os.path.join(org_data_folder,'amplitude_bscan.npy'))
    
    t = (np.arange(org_data.shape[1]-4)-stimulus_index)*bscan_interval

    data = []
    coords = []
    for rowidx in range(org_data.shape[0]):
        row = org_data[rowidx,:]
        rowcoords = row[:4].astype(int)
        rowdata = row[4:].astype(float)
        data.append(rowdata)
        coords.append(rowcoords)
        
    data = np.array(data)
    mdata = np.mean(data,axis=0)

    plt.figure(figsize=(6,6))
    plt.plot(t,mdata)

    plt.figure()
    plt.imshow(dB(bscan),clim=dB_clims,cmap='gray')
    for x1,y1,x2,y2 in coords:
        plt.plot(x1,y1,'ro',markersize=4)
        plt.plot(x2,y2,'ro',markersize=4)

    
    plt.show()
