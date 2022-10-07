from octoblob import functions as blobf
import numpy as np
from matplotlib import pyplot as plt
import time
import numba

try:
    import multiprocessing
    use_multiprocessing = True
except ImportError:
    use_multiprocessing = False
    
use_multiprocessing = False

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

show = True

metrics = [blobf.iq_max,blobf.iq_maxes,blobf.gradient_mean,blobf.gradient_median,blobf.average_aline_contrast,blobf.sharpness]
results = []
for metric in metrics:
    results.append(np.ones((sy,sx))*np.nan)


t0 = time.time()
if not use_multiprocessing:
    for idx3,c3 in enumerate(grid3):
        print('%d of %d rows'%(idx3,len(grid3)))
        for idx2,c2 in enumerate(grid2):
            temp = blobf.dispersion_compensate(spectra,[c3,c2])
            temp = np.abs(np.fft.fft(temp,axis=0))
            temp = temp[z1:z2,:]
            for metric,result in zip(metrics,results):
                result[idx3,idx2] = metric(temp)
else:
    tups = []
    for idx3,c3 in enumerate(grid3):
        for idx2,c2 in enumerate(grid2):
            tups.append((c3,c2,idx3,idx2))
    def f(tup):
        c3,c2,idx3,idx2 = tup
        temp = blobf.dispersion_compensate(spectra,[c3,c2])
        temp = np.abs(np.fft.fft(temp,axis=0))
        temp = temp[z1:z2,:]
        for metric,result in zip(metrics,results):
            result[idx3,idx2] = metric(temp)
    p = multiprocessing.Pool(16)
    p.map(f,tups)
elapsed = time.time()-t0
print('use_multiprocessing: %s, elapsed: %0.3f'%(use_multiprocessing,elapsed))


if show:
    plt.figure()
    for k in range(1,len(metrics)+1):
        plt.subplot(2,3,k)
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
    plt.show()
# if show:
#     plt.savefig('results.png',dpi=300)
#     plt.show()



# if show:
#     plt.clf()
#     for k in range(1,len(metrics)+1):
#         plt.subplot(2,3,k)
#         plt.cla()
#         plt.imshow(results[k-1],extent=[np.min(grid2),np.max(grid2),
#                                         np.min(grid3),np.max(grid3)],
#                    interpolation='none',cmap='jet')
#         plt.colorbar()
#         plt.axis('auto')
#         plt.plot(*manual_dispersion_coefficients[::-1],'k.',alpha=.5)
#         plt.title(metrics[k-1].__doc__)
#         if k in [1,4]:
#             plt.ylabel('c3')
#         else:
#             plt.yticks([])
#         if k in [4,5,6]:
#             plt.xlabel('c2')
#         else:
#             plt.xticks([])
#     if idx2%20==0:
#         plt.pause(.00001)
