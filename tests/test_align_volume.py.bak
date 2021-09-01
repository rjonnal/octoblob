import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from octoblob.registration_tools import rigid_register

if len(sys.argv)<2:
    print('Usage: python test_project_enface.py input_directory')
    sys.exit()

## Three-stage alignment:
## First, B-scans are shifted in fast (x) and depth (z) dimensions, as follows. Pairwise
## cross-correlations are used to estimate velocities (dx/dt and dz/dt), and these
## are integrated to estimate position. This approach is susceptible to accumulative
## error. Residual misregistration (even sub-pixel) has non-zero RMS, and thus the
## total position error accumulates in a random walk.
## In the second stage, B-scans are aligned in z by fitting of a polynomial to the brightest
## points in depth in the fast (x) projection. The fitting is weighted by the brightness
## of those points.
## The third stage is just like the second, but in the slow (y) projection.


input_directory = sys.argv[1]
output_directory = os.path.join(input_directory,'registered')
info_directory = os.path.join(output_directory,'info')

crop_top = 10
crop_bottom = 10

fitting_order = 3

try:
    os.makedirs(output_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(output_directory)
    except Exception as e:
        print('%s exists; using existing directory')

try:
    os.makedirs(info_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(info_directory)
    except Exception as e:
        print('%s exists; using existing directory')

        
assert os.path.exists(input_directory)

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


dx = [0]
dz = [0]
xcmax = [0]

def norm(im):
    return (im - im.mean())/im.std()

try:
    dx = np.loadtxt(os.path.join(info_directory,'dx_1.txt')).astype(np.int)
    dz = np.loadtxt(os.path.join(info_directory,'dz_1.txt')).astype(np.int)
    xcmax = np.loadtxt(os.path.join(info_directory,'xcmax_1.txt'))
except Exception as e:
    ref = vol[0,:,:]
    for k in range(1,vol.shape[0]):
        tar = vol[k,:,:]
        x,y,xc=rigid_register(norm(tar)[crop_top:-crop_bottom,:],norm(ref)[crop_top:-crop_bottom,:],max_shift=100)
        print(k,x,y)
        dx.append(x)
        dz.append(y)
        xcmax.append(xc.max()/np.prod(tar.shape))
        ref = tar
    dx = np.array(dx).astype(np.int)
    dz = np.array(dz).astype(np.int)
    xcmax = np.array(xcmax)
    
    np.savetxt(os.path.join(info_directory,'dx_1.txt'),dx)
    np.savetxt(os.path.join(info_directory,'dz_1.txt'),dz)
    np.savetxt(os.path.join(info_directory,'xcmax_1.txt'),xcmax)


dx[np.where(np.abs(dx)>20)] = 0

plt.figure()
plt.plot(dx,label='dx/dt')
plt.plot(dz,label='dz/dt')
plt.title('pairwise displacements')




dz = np.cumsum(dz)
dx = np.cumsum(dx)
    
plt.figure()
plt.plot(dx,label='x')
plt.plot(dz,label='y')
plt.title('accumulated displacement (position)')

plt.figure()
plt.imshow(vol.mean(2).T,cmap='gray')
plt.autoscale(False)
dz_line = dz+vol.shape[1]//2
plt.plot(dz_line,color='y')

dz = -dz
dx = -dx
    
dz = dz - dz.min()
dx = dx - dx.min()

aligned_vol = np.zeros((vol.shape[0],vol.shape[1]+dz.max(),vol.shape[2]+dx.max()))

for k in range(vol.shape[0]):
    aligned_vol[k,dz[k]+crop_top:dz[k]+vol.shape[1]-crop_bottom,dx[k]:dx[k]+vol.shape[2]] = vol[k,crop_top:-crop_bottom,:]


# don't need vol anymore, so let's get rid of it
vol = aligned_vol

x_projection = vol.mean(2).T


z_peaks = np.argmax(x_projection,axis=0)
peaks = np.max(x_projection,axis=0)

y = np.arange(vol.shape[0])
bright_polynomial = np.polyfit(y,z_peaks,fitting_order,w=peaks)
fitted_z = np.polyval(bright_polynomial,y)

plt.figure()
plt.imshow(x_projection,cmap='gray')
plt.title('step 2: fast projection\nwith fit')
plt.plot(y,z_peaks,'y.',label='peaks')
plt.plot(y,fitted_z,'r--',label='fit')
plt.legend()

fitted_z = -fitted_z
dz = np.round(fitted_z).astype(np.int)
dz = dz-dz.min()
np.savetxt(os.path.join(info_directory,'dz_2.txt'),dz)

aligned_vol = np.zeros((vol.shape[0],vol.shape[1]+dz.max(),vol.shape[2]))

for k in range(vol.shape[0]):
    print(dz[k])
    aligned_vol[k,dz[k]:dz[k]+vol.shape[1],:] = vol[k,:,:]


# don't need vol anymore, so get rid of it again
vol = aligned_vol
x_projection = vol.mean(2).T

plt.figure()
plt.imshow(x_projection,cmap='gray')
plt.title('second stage corrected')

y_projection = vol.mean(0)

z_peaks = np.argmax(y_projection,axis=0)
peaks = np.max(y_projection,axis=0)
x = np.arange(vol.shape[2])
bright_polynomial = np.polyfit(x,z_peaks,fitting_order,w=peaks)
fitted_z = np.polyval(bright_polynomial,x)

plt.figure()
plt.imshow(y_projection,cmap='gray')
plt.title('step 3: slow projection\nwith fit')
plt.plot(x,z_peaks,'y.',label='peaks')
plt.plot(x,fitted_z,'r--',label='fit')
plt.legend()

fitted_z = -fitted_z
dz = np.round(fitted_z).astype(np.int)
dz = dz-dz.min()
np.savetxt(os.path.join(info_directory,'dz_3.txt'),dz)

aligned_vol = np.zeros((vol.shape[0],vol.shape[1]+dz.max(),vol.shape[2]))

for k in range(vol.shape[2]):
    print(dz[k])
    aligned_vol[:,dz[k]:dz[k]+vol.shape[1],k] = vol[:,:,k]

vol = aligned_vol
y_projection = vol.mean(0)
    
plt.figure()
plt.imshow(y_projection,cmap='gray')
plt.title('third stage corrected')


plt.show()


for k in range(vol.shape[0]):
    outfn = os.path.join(output_directory,os.path.split(flist[k])[1])
    print('Saving registered frame to %s.'%outfn)
    np.save(outfn,vol[k,:,:])

