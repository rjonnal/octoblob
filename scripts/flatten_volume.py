import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from octoblob.registration_tools import rigid_register
import scipy.signal as sps

flatten_x = False

# If the python command has less than 2 arguments, print the instructions and call it quits.
if len(sys.argv)<2:
    print('Usage: python test_project_enface.py input_directory')
    sys.exit()

## Three-stage alignment:
## First, B-scans are shifted in fast (x) and depth (z) dimensions, as follows. Pairwise
## cross-correlations are used to estimate velocities (dx/dt and dz/dt), and these
## are integrated to estimate position. This approach is susceptible to accumulative
## error. Residual misregistration (even sub-pixel) has non-zero RMS, and thus the
## total position error accumulates in a random walk.
## In the second stage, B-scans are flattened in z by fitting of a polynomial to the brightest
## points in depth in the fast (x) projection. The fitting is weighted by the brightness
## of those points.
## The third stage is just like the second, but in the slow (y) projection.

# The second argument you specify in the command line is the input directory.
input_directory = sys.argv[1]

#  Create subfolder to the input directory to save the flattened data and not overwrite data.
output_directory = os.path.join(input_directory,'flattened')

# Another subfolder for whatever info we want to store (dx/dx/xcorr).
info_directory = os.path.join(output_directory,'info')

# Volume cache directory
vcache_directory = os.path.join(input_directory,'volume')

#  Cropping the B-scans 
crop_top = 10
crop_bottom = 10

# Choose the polynomial fitting order
fitting_order = 3

# should we redo the registration each time this is run, or try to read
# cached values?
redo = True

# cache the volume so it doesn't always have to be reloaded?
cache_volume = True

# what is the maximum allowable shift between adjacent B-scans?
maximum_allowable_shift = 10

gradient_smoothing_kernel = 0
gradient_smoothing_kernel = np.ones((5,13))
gradient_threshold = 50
show_segmentation = False

assert os.path.exists(input_directory)

# Checks if the directory already exists and  uses the existing one.
try:
    os.makedirs(output_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(output_directory)
    except Exception as e:
        print('%s exists; using existing directory')

# Same thing for the info folder
try:
    os.makedirs(info_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(info_directory)
    except Exception as e:
        print('%s exists; using existing directory')

# Same thing for the info folder
try:
    os.makedirs(vcache_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(vcache_directory)
    except Exception as e:
        print('%s exists; using existing directory')


volume_loaded = False
if cache_volume:
    try:
        vol = np.load(os.path.join(vcache_directory,'volume.npy'))
        volume_loaded = True
    except Exception:
        pass
# List the .npy files in the folder and sort them just in case.
flist = glob.glob(os.path.join(input_directory,'*.npy'))
flist.sort()

if not volume_loaded:
    # Create an empty list to store the loaded B-scans
    vol = []

    for idx,f in enumerate(flist):
        arr = np.load(f)

        # in case data are complex:
        arr = np.abs(arr)
        nfiles = len(flist)
        # BM-scans are averaged before any alignment:
        try:
            assert len(arr.shape)==2
        except AssertionError as ae:
            print('Averaging stack in slow/BM direction; %d of %d.'%(idx+1,nfiles))
            arr = arr.mean(2)

        vol.append(arr)

    # Cast volume to be a 3D array.
    vol = np.array(vol)

    if cache_volume:
        np.save(os.path.join(vcache_directory,'volume.npy'),vol)


# Create lists to store the x, z, and correlations; initialize them
# with values--these are the shifts between the first B-scan and
# an imagined previous B-scan, so they might as well be zero.
dx = [0]
dz = [0]
xcmax = [0]

# Define a normalization function that can be used to do normalized
# cross-correlation (makes comparisons of the xc peaks meaningful;
# without normalization, brighter B-scans will always appear to be
# more highly correlated than dimmer ones, regardless of the true
# feature correspondence)
def norm(im):
    return (im - im.mean())/im.std()


# if we already ran the registration before, try to load the previous
# coordinates
try:
    # raise an AssertionException if redo=True:
    assert redo==False
    
    dx = np.loadtxt(os.path.join(info_directory,'dx_1.txt')).astype(np.int)
    dz = np.loadtxt(os.path.join(info_directory,'dz_1.txt')).astype(np.int)
    xcmax = np.loadtxt(os.path.join(info_directory,'xcmax_1.txt'))
    
except Exception as e:
    # Set first averaged B-scan as reference, for the first pairwise
    # comparison; at the end of the loop, we'll reset the reference ref
    # to be the one we just registered
    ref = vol[0,:,:]
    
    for k in range(1,vol.shape[0]):
        # On each iteration of the loop, the registration target (tar)
        # is the kth B-scan and the reference (ref) is the (k-1)th
        tar = vol[k,:,:]
        
        # Do rigid register  using normalized BM-scans with cropping included and limiting the shift to100 pixels (1/4 of a B-scan). The function outputs shift in dx and dz plus the xcorr value.
        x,y,xc=rigid_register(norm(tar)[crop_top:-crop_bottom,:],norm(ref)[crop_top:-crop_bottom,:],max_shift=maximum_allowable_shift,diagnostics=False)

        if not flatten_x:
            x = 0
        print(k,x,y)
        
        # Add the x/y displacements to the dx/dz vectors.
        dx.append(x)
        dz.append(y)
        xcmax.append(xc.max()/np.prod(tar.shape))
        
        # The current target BM-scan becomes the  reference. No universal reference frame .
        ref = tar
    
    # Save the shifts in the array as well as the xcorr peak value
    dx = np.array(dx).astype(np.int)
    dz = np.array(dz).astype(np.int)
    xcmax = np.array(xcmax)

    plt.figure()
    shift_mag = np.sqrt(dz**2+dx**2)
    shift_mag_mean = np.mean(shift_mag)
    shift_mag_std = np.std(shift_mag)
    theoretical_limit = shift_mag_mean+3*shift_mag_std
    plt.hist(shift_mag,bins=12)
    plt.axvline(maximum_allowable_shift,color='g')
    plt.text(maximum_allowable_shift,2,'maximum_allowable_shift',color='g',ha='right',va='bottom')
    plt.axvline(theoretical_limit,color='r')
    plt.text(theoretical_limit,plt.gca().get_ylim()[1],'statistical limit\n$\overline{s}+3\sigma$',va='top',color='r')
    plt.title('shift magnitude dist stage 1')
    
    # Write down the values to the text files in the info subfolder.
    np.savetxt(os.path.join(info_directory,'dx_1.txt'),dx)
    np.savetxt(os.path.join(info_directory,'dz_1.txt'),dz)
    np.savetxt(os.path.join(info_directory,'xcmax_1.txt'),xcmax)

# Remove this; it's fixed in a more sensible way with maximum_allowable_shift
# If there is a shift that is more than 20 pixels we set it to zero?
# dx[np.where(np.abs(dx)>20)] = 0

# This plots the displacement between the B-scans.
plt.figure()
plt.plot(dx,label='dx/dt')
plt.plot(dz,label='dz/dt')
plt.title('pairwise displacements')

# Cumulative displacement over the B-scan pairs
dz = np.cumsum(dz)
dx = np.cumsum(dx)
 
 # Does the second plot that the  algorithm outputs.
plt.figure()
plt.plot(dx,label='x')
plt.plot(dz,label='y')
plt.title('accumulated displacement (position)')

#  Average the volume in fast axis direction 
plt.figure()
plt.imshow(vol.mean(2).T,cmap='gray')
plt.autoscale(False)
dz_line = dz+vol.shape[1]//2 #Get the  dz pixel shift  and add  an offset so that it's not on top of the BM-scan.
plt.plot(dz_line,color='y')

# we need to invert these values because of the order of tar and ref in the
# rigid register function:
dz = -dz
dx = -dx

# shift the coordinates so that they have 0 min, for ease of indexing
# into the new volume; it doesn't matter since the absolute coordinate
# in the new volume is meaningless
dz = dz - dz.min()
dx = dx - dx.min()

# allocate memory for flattened volume  that is slightly larger than original volume due to max. shifts.
flattened_vol = np.zeros((vol.shape[0],vol.shape[1]+dz.max(),vol.shape[2]+dx.max()))

# Here each BM-scan is shifted according to the rigid_register function output.
for k in range(vol.shape[0]):
    flattened_vol[k,dz[k]+crop_top:dz[k]+vol.shape[1]-crop_bottom,dx[k]:dx[k]+vol.shape[2]] = vol[k,crop_top:-crop_bottom,:]

# don't need vol anymore, so let's get rid of it
vol = flattened_vol

# let's keep the depth derivative too in case gradients are more useful
dvol = np.diff(flattened_vol,axis=1) # the 1st dimension is depth

working_vol = -dvol

try:
    dz = np.loadtxt(os.path.join(info_directory,'dz_2.txt')).astype(np.int)
except Exception as e:
    # Fast axis projection
    x_projection = working_vol.mean(2).T


    # normalize the A-scans so we can use a single gradient threshold for all
    xstd = np.std(x_projection,axis=0)
    xmean = np.mean(x_projection,axis=0)
    x_projection = (x_projection-xmean)/xstd

    # do a bit of smoothing to reduce spurious gradients
    x_projection = sps.fftconvolve(x_projection,gradient_smoothing_kernel,mode='same')
    x_projection_profile = np.mean(x_projection,axis=1)
    plt.figure()
    plt.plot(x_projection_profile)


    left = x_projection_profile[:-2]
    center = x_projection_profile[1:-1]
    right = x_projection_profile[2:]
    peaks = np.where(np.logical_and(center>left,center>right))[0]+1
    peaks = [p for p in peaks if x_projection_profile[p]>gradient_threshold]

    brm_idx = peaks[-1]
    start_idx = np.argmax(x_projection[brm_idx,:])
    
    print(peaks)

    finished = False
    sy,sx = x_projection.shape

    brm_coords = [(start_idx,brm_idx)]
    right_x = start_idx+1
    left_x = start_idx-1
    right_z = brm_idx
    left_z = brm_idx

    if show_segmentation:
        plt.figure()
        plt.imshow(x_projection,aspect='auto',cmap='gray')
        plt.plot(start_idx,brm_idx,'go')
        plt.colorbar()

    while not finished:
        if right_x<sx:
            candidates=x_projection[right_z-2:right_z+3,right_x]
            dz = np.argmax(candidates)-2
            right_z = right_z+dz
            brm_coords.append((right_x,right_z))
            right_x = right_x+1
            if show_segmentation:
                plt.plot(right_x,right_z,'g.')
            
        if left_x>-1:
            candidates=x_projection[left_z-2:left_z+3,left_x]
            dz = np.argmax(candidates)-2
            left_z = left_z+dz
            brm_coords.append((left_x,left_z))
            left_x = left_x-1
            if show_segmentation:
                plt.plot(left_x,left_z,'r.')

        finished = right_x==sx and left_x==-1
        if show_segmentation:
            plt.pause(.001)

    brm_coords.sort(key = lambda x: x[0])
    print(brm_coords)

    fitted_z = [t[1] for t in brm_coords]
    fitted_z = np.array(fitted_z)

    if False:
        ## This is the part where I think we  could do a quick fix. ##
        # 1. crop x_projection before finding z_peaks and peaks
        # 2. Add the cropped  "offset" to correctly locate the peak in the "full" B-scan. 

        # Obtain the  indices of the maximum values  from the fast axis projection.
        z_peaks = np.argmax(x_projection,axis=0)

        # Obtain the maximum values.
        peaks = np.max(x_projection,axis=0)

        # Create a vector that is the same size as the  flattened volume.
        y = np.arange(working_vol.shape[0])

        # Fit polynomial to the  peak indeces with weight on the peak values themselves. Fitting order is set in the beginning, currently 3.
        bright_polynomial = np.polyfit(y,z_peaks,fitting_order,w=peaks)

        # According to documentation, new version is  np.polynomial. Get the values from the fit for each y indices (The location where the peak "should be").
        fitted_z = np.polyval(bright_polynomial,y)

        # This one plots the fast axis  projection with the detected peaks and 3rd degree polynomial fit with dashed line.
        plt.figure()
        plt.imshow(x_projection,cmap='gray')
        plt.title('step 2: fast projection\nwith fit')
        plt.plot(y,z_peaks,'y.',label='peaks')
        plt.plot(y,fitted_z,'r--',label='fit')
        plt.legend()

        # Not fully sure what's happening here. Probably the same as before to get the thing to start from zero for easier next step.
    fitted_z = -fitted_z
    dz = np.round(fitted_z).astype(np.int)
    dz = dz-dz.min()
    np.savetxt(os.path.join(info_directory,'dz_2.txt'),dz)

# Allocate memory for the final volume taking into account the maximum dz value from the fit.
flattened_vol = np.zeros((vol.shape[0],vol.shape[1]+dz.max(),vol.shape[2]))

# Do the flattening of volume and save it to the new flattened_vol
for k in range(vol.shape[0]):
    print(dz[k])
    flattened_vol[k,dz[k]:dz[k]+vol.shape[1],:] = vol[k,:,:]

# don't need vol anymore, so get rid of it again
vol = flattened_vol

# let's keep the depth derivative too in case gradients are more useful
dvol = np.diff(flattened_vol,axis=1) # the 1st dimension is depth

working_vol = -dvol

try:
    dz = np.loadtxt(os.path.join(info_directory,'dz_3.txt')).astype(np.int)
except:
    x_projection = vol.mean(2).T # Do a projection through fast axis again, this time the volume is flattened and plot it.
    
    plt.figure()
    plt.imshow(x_projection,cmap='gray')
    plt.title('second stage corrected')

    #  Do a projection through slow axis 
    y_projection = vol.mean(0)

    # normalize the A-scans so we can use a single gradient threshold for all
    ystd = np.std(y_projection,axis=0)
    ymean = np.mean(y_projection,axis=0)
    y_projection = (y_projection-ymean)/ystd

    # do a bit of smoothing to reduce spurious gradients
    y_projection = sps.fftconvolve(y_projection,gradient_smoothing_kernel,mode='same')
    y_projection_profile = np.mean(y_projection,axis=1)
    plt.figure()
    plt.plot(y_projection_profile)


    left = y_projection_profile[:-2]
    center = y_projection_profile[1:-1]
    right = y_projection_profile[2:]
    peaks = np.where(np.logical_and(center>left,center>right))[0]+1
    peaks = [p for p in peaks if y_projection_profile[p]>gradient_threshold]

    brm_idx = peaks[-1]
    start_idx = np.argmax(y_projection[brm_idx,:])
    
    print(peaks)

    finished = False
    sy,sx = y_projection.shape

    brm_coords = [(start_idx,brm_idx)]
    right_x = start_idx+1
    left_x = start_idx-1
    right_z = brm_idx
    left_z = brm_idx

    if show_segmentation:
        plt.figure()
        plt.imshow(y_projection,aspect='auto',cmap='gray')
        plt.plot(start_idx,brm_idx,'go')
        plt.colorbar()

    while not finished:
        if right_x<sx:
            candidates=y_projection[right_z-2:right_z+3,right_x]
            dz = np.argmax(candidates)-2
            right_z = right_z+dz
            brm_coords.append((right_x,right_z))
            right_x = right_x+1
            if show_segmentation:
                plt.plot(right_x,right_z,'g.')
            
        if left_x>-1:
            candidates=y_projection[left_z-2:left_z+3,left_x]
            dz = np.argmax(candidates)-2
            left_z = left_z+dz
            brm_coords.append((left_x,left_z))
            left_x = left_x-1
            if show_segmentation:
                plt.plot(left_x,left_z,'r.')

        finished = right_x==sx and left_x==-1
        if show_segmentation:
            plt.pause(.001)

    brm_coords.sort(key = lambda x: x[0])
    print(brm_coords)

    fitted_z = [t[1] for t in brm_coords]
    fitted_z = np.array(fitted_z)


    if False:
        # Same as above, find the brightest  pixel indices and values and then do polynomial fit.
        z_peaks = np.argmax(y_projection,axis=0)
        peaks = np.max(y_projection,axis=0)
        x = np.arange(vol.shape[2])
        bright_polynomial = np.polyfit(x,z_peaks,fitting_order,w=peaks)
        fitted_z = np.polyval(bright_polynomial,x)

        # Plot the  slow axis projection with the  peaks and poly fit.
        plt.figure()
        plt.imshow(y_projection,cmap='gray')
        plt.title('step 3: slow projection\nwith fit')
        plt.plot(x,z_peaks,'y.',label='peaks')
        plt.plot(x,fitted_z,'r--',label='fit')
        plt.legend()

    # This is that zero trick again before saving the values and creating the flattened_vol for the last time.
    fitted_z = -fitted_z
    dz = np.round(fitted_z).astype(np.int)
    dz = dz-dz.min()
    np.savetxt(os.path.join(info_directory,'dz_3.txt'),dz)

flattened_vol = np.zeros((vol.shape[0],vol.shape[1]+dz.max(),vol.shape[2]))

# Again, shift the  slow axis averaged A-scans  according to the fit params.
for k in range(vol.shape[2]):
    print(dz[k])
    flattened_vol[:,dz[k]:dz[k]+vol.shape[1],k] = vol[:,:,k]

# Getting rid of the volume again x 3.
vol = flattened_vol

# Plot the slow axis projection
y_projection = vol.mean(0)
plt.figure()
plt.imshow(y_projection,cmap='gray')
plt.title('third stage corrected')

# This will make all the plots above to be plotted on the screen.
plt.show()

# This part will only  execute itself after the plots are closed on the screen.
for k in range(vol.shape[0]):
    outfn = os.path.join(output_directory,os.path.split(flist[k])[1])
    print('Saving registered frame to %s.'%outfn)
    np.save(outfn,vol[k,:,:])

