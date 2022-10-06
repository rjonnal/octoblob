from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt

manual_dispersion_coefficients = [-1.460e-08, -2.900e-07]
dispersion_steps = [1e-9,1e-8]

spectra = blobf.load_spectra('spectra_00100.npy')
spectra = blobf.dc_subtract(spectra)

z1 = 1050
z2 = 1200

grid3 = np.arange(-2*abs(manual_dispersion_coefficients[0]),abs(manual_dispersion_coefficients[0]),dispersion_steps[0])
grid2 = np.arange(-2*abs(manual_dispersion_coefficients[1]),abs(manual_dispersion_coefficients[1]),dispersion_steps[1])

sy = len(grid3)
sx = len(grid2)

# Image quality metrics
def iq_max(im):
    """Image max"""
    return np.max(im)

def iq_maxes(im):
    """Mean of brightest\n1 pct of pixels"""
    temp = im.ravel()
    N = round(len(temp)*0.01)
    temp = np.partition(-temp, N)
    result = -temp[:N]
    return np.mean(result)

def gradient_mean(im):
    """Mean of absolute\nz-derivative"""
    return np.mean(np.abs(np.diff(im,axis=0)))

def gradient_median(im):
    """Median of absolute\nz-derivative"""
    return np.mean(np.abs(np.diff(im,axis=0)))

def average_aline_contrast(im):
    """Mean of A-scan\nMichelson contrast""" 
    x = np.max(im,axis=0)
    n = np.min(im,axis=0)
    return np.mean((x-n)/(x+n))

def sharpness(im):
    """Image sharpness"""
    return np.sum(im**2)/(np.sum(im)**2)

show = True

metrics = [iq_max,iq_maxes,gradient_mean,gradient_median,average_aline_contrast,sharpness]
results = []
for metric in metrics:
    results.append(np.ones((sy,sx))*np.nan)

for idx3,c3 in enumerate(grid3):
    for idx2,c2 in enumerate(grid2):
        print(c3,c2)
        temp = blobf.dispersion_compensate(spectra,[c3,c2])
        temp = np.abs(np.fft.fft(temp,axis=0))
        temp = temp[z1:z2,:]
        for metric,result in zip(metrics,results):
            result[idx3,idx2] = metric(temp)
        if show:
            plt.clf()
            for k in range(1,len(metrics)+1):
                plt.subplot(2,3,k)
                plt.cla()
                plt.imshow(results[k-1],extent=[np.min(grid2),np.max(grid2),
                                                np.min(grid3),np.max(grid3)],
                           interpolation='none',cmap='jet')
                plt.colorbar()
                plt.axis('auto')
                plt.plot(*manual_dispersion_coefficients[::-1],'k.',alpha=.5)
                plt.title(metrics[k-1].__doc__)
                if k in [1,4]:
                    plt.ylabel('c3')
                else:
                    plt.yticks([])
                if k in [4,5,6]:
                    plt.xlabel('c2')
                else:
                    plt.xticks([])
            if idx2%20==0:
                plt.pause(.00001)

if show:
    plt.savefig('results.png',dpi=300)
    plt.show()
