import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from .ticktock import tick, tock
import scipy.ndimage as spn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("volume_tools_debug.log"),
        logging.StreamHandler()
    ]
)

# for loading M-scan series as volumes, we average the abs along this dimension:
M_SCAN_DIMENSION = 2

class Space:

    def __init__(self,initialization_volume):
        self.n_slow,self.n_depth,self.n_fast = initialization_volume.shape
        logging.info('Initializing coordinate space with shape %d x %d x %d (slow x depth x fast).'%(self.n_slow,self.n_depth,self.n_fast))
        self.entries = []
        self.entries.append((initialization_volume,(0,0,0)))


    def put(self,volume,coordinates):
        pass

def norm(im):
    return (im - im.mean())/im.std()

def nxc3d(ref,tar,downsample=1,diagnostics=False):
    ref = norm(ref)
    tar = norm(tar)

    n_slow,n_depth,n_fast = ref.shape

    logging.info('Registering volumes of shape %dx%dx%d (slow x depth x fast).'%(n_slow,n_depth,n_fast))
    t0 = tick()
    
    if diagnostics:
        plt.figure()
        plt.subplot(4,2,1)
        plt.imshow(np.mean(ref,axis=1))
        plt.subplot(4,2,2)
        plt.imshow(np.mean(tar,axis=1))
        plt.subplot(4,2,5)
        plt.imshow(np.mean(ref,axis=0))
        plt.subplot(4,2,6)
        plt.imshow(np.mean(tar,axis=0))
        
    ref = ref[::downsample,::downsample,::downsample]
    tar = tar[::downsample,::downsample,::downsample]

    if diagnostics:
        plt.subplot(4,2,3)
        plt.imshow(np.mean(ref,axis=1))
        plt.subplot(4,2,4)
        plt.imshow(np.mean(tar,axis=1))
        plt.subplot(4,2,7)
        plt.imshow(np.mean(ref,axis=0))
        plt.subplot(4,2,8)
        plt.imshow(np.mean(tar,axis=0))

    
    rsx,rsz,rsy = ref.shape
    tsx,tsz,tsy = tar.shape
    
    sx = max(rsx,tsx)
    sz = max(rsz,tsz)
    sy = max(rsy,tsy)
    
    t0 = tick()
    s = (sx,sz,sy)
    fref = np.fft.fftn(ref,s=s)
    ftar = np.fft.fftn(tar,s=s)
    dt = tock(t0)
    logging.info('Registration took %0.3f sec with downsampling of %d.'%(dt,downsample))
    
    return np.real(np.fft.ifftn(fref*np.conj(ftar)))


class Coordinates:
    """A Coordinates object keeps track of the 3D coordinates for each A-scan in a Volume object."""
    def __init__(self,n_slow,n_depth,n_fast):
        self.x,self.y = np.meshgrid(np.arange(n_fast),np.arange(n_slow))
        self.z = np.zeros(self.x.shape,dtype=np.int)
        self.sy,self.sx = self.z.shape
        
    def move_x(self,dx,y1=None,y2=None,x1=None,x2=None):
        if y1 is None or y2 is None:
            y1 = 0
            y2 = self.sy
        if x1 is None or x2 is None:
            x1 = 0
            x2 = self.sx

        self.x[y1:y2,x1:x2]+=dx

    def move_y(self,dy,y1=None,y2=None,x1=None,x2=None):
        if y1 is None or y2 is None:
            y1 = 0
            y2 = self.sy
        if x1 is None or x2 is None:
            x1 = 0
            x2 = self.sx

        self.y[y1:y2,x1:x2]+=dy

    def move_z(self,dz,y1=None,y2=None,x1=None,x2=None):
        if y1 is None or y2 is None:
            y1 = 0
            y2 = self.sy
        if x1 is None or x2 is None:
            x1 = 0
            x2 = self.sx

        self.z[y1:y2,x1:x2]+=dz


class Boundaries:

    def __init__(self,y1,y2,z1,z2,x1,x2):
        sy = y2-y1
        sz = z2-z1
        sx = x2-x1
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        self.shape = (sy,sz,sx)
        
        
class Volume:

    def __init__(self,bscan_directory,initial_coords=None,cache=True,diagnostics=False):
        t0 = tick()
        
        self.bscan_directory = bscan_directory
        self.bscan_filenames = sorted(glob.glob(os.path.join(self.bscan_directory,'*.npy')))

        # determine volume shape from file list length and sample B-scan:
        self.n_slow = len(self.bscan_filenames)
        temp = np.load(self.bscan_filenames[0])
        self.n_depth,self.n_fast = temp.shape

        # set default coordinates:
        self.coordinates = Coordinates(self.n_slow,self.n_depth,self.n_fast)
        if initial_coords is not None:
            self.coordinates.move_x(initial_coords[2])
            self.coordinates.move_y(initial_coords[0])
            self.coordinates.move_z(initial_coords[1])

        self.cache = cache
        self.cache_dir = os.path.join(self.bscan_directory,'volume')
        os.makedirs(self.cache_dir,exist_ok=True)

    def build_volume(self,diagnostics=False):
        t0 = tick()
        logging.info('Building volume in %s.'%self.bscan_directory)
        temp = np.load(self.bscan_filenames[0])
        dtype = temp.dtype
        is_stack = len(temp.shape)>2

        volume = []
        for rf in self.bscan_filenames:
            temp = np.load(rf)
            temp = np.abs(temp)
            if is_stack:
                temp = temp.mean(M_SCAN_DIMENSION)
                
            if diagnostics:
                plt.cla()
                plt.imshow(temp,cmap='gray')
                plt.pause(.1)

            volume.append(temp)
        volume = np.array(volume,dtype=dtype)
        logging.info('Done; took %0.3f sec.'%tock(t0))
        return volume
    
        
    def get_volume(self,diagnostics=False):
        
        if self.cache:
            vol_fn = os.path.join(self.cache_dir,'volume.npy')

            if os.path.exists(vol_fn):
                t0 = tick()
                out_vol = np.load(vol_fn)
                logging.info('Using cached version at %s; took %0.3f sec.'%(vol_fn,tock(t0)))
                return out_vol

            else:
                out_vol = self.build_volume(diagnostics=diagnostics)
                return out_vol

        else:
            out_vol = self.build_volume(diagnostics=diagnostics)
            return out_vol

    def move(self,shifts):
        self.coordinates.move_y(shifts[0])
        self.coordinates.move_z(shifts[1])
        self.coordinates.move_x(shifts[2])


    def get_block(self,b,volume=None):
        # given boundaries b, return a subvolume of my volume
        # using my coordinates
        # use case:
        # The registration script will step through the reference coordinate system,
        # e.g., by slices, using unshifted coordinates (e.g., bscans 0-9, 10-19, etc.).
        # For each of these, it will create a boundary object, and pass it to this
        # object. This object will then have to use its coordinates object to determine
        # where to pull the block from.
        # One problem is that the requested block may wind up being the wrong size, if
        # this object's coordinates say so. For now, we'll arbitrarily pad or truncate
        # the block to match the requested size, but this may have to be revisited later.
        # (This may only happen if in a multiscale approach, the block size is increased.
        # If the block size is always reduced, it shouldn't?)
        block = np.ones(b.shape)*np.nan
        xc = np.copy(self.coordinates.x)
        yc = np.copy(self.coordinates.y)
        zc = np.copy(self.coordinates.z)

        if volume is None:
            volume = self.get_volume()

        # Ugh this is going to be so slow. I'm too stupid to vectorize it.
        for yput in range(b.y1,b.y2):
            for xput in range(b.x1,b.x2):
                yget,xget = np.where((xc==xput)*(yc==yput))
                if len(yget):
                    zget1 = zc[yput,xput]
                    zget2 = zget1+self.n_depth
                    zput1 = 0
                    zput2 = self.n_depth
                    while zget1<0:
                        zget1+=1
                        zput1+=1
                    while zget2>self.n_depth:
                        zget2-=1
                        zput2-=1
                    block[yput,zget1:zget2,xput] = volume[yget,zput1:zput2,xget]
        print(zget1,zget2,zput1,zput2)

        plt.figure()
        plt.imshow(np.abs(block[b.shape[0]//2,:,:]))

    def register_to(self,reference_volume,downsample=1,diagnostics=False):
        nxc = nxc3d(reference_volume.get_volume(),self.get_volume(),downsample=downsample,diagnostics=diagnostics)
        reg_coords = list(np.unravel_index(np.argmax(nxc),nxc.shape))
        for idx in range(len(nxc.shape)):
            if reg_coords[idx]>nxc.shape[idx]//2:
                reg_coords[idx] = reg_coords[idx]-nxc.shape[idx]

        if diagnostics:
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(nxc[reg_coords[0],:,:])
            plt.subplot(1,3,2)
            plt.imshow(nxc[:,reg_coords[1],:])
            plt.subplot(1,3,3)
            plt.imshow(nxc[:,:,reg_coords[2]])
            plt.show()

        full_reg_coords = [rc*downsample for rc in reg_coords]
        self.move(full_reg_coords)

                
class VolumeSeries:

    def __init__(self,summing_function=np.abs):
        self.volumes = []
        self.summing_function = summing_function

    def __getitem__(self,n):
        return self.volumes[n]

    def add(self,volume):
        self.volumes.append(volume)


    def render(self,diagnostics=False):
        large_integer = 10000000000
        ymin,zmin,xmin = large_integer,large_integer,large_integer

        # find the global min for each dimension, for min-subtraction
        for v in self.volumes:
            if v.coordinates.z.min()<zmin:
                zmin = v.coordinates.z.min()
            if v.coordinates.y.min()<ymin:
                ymin = v.coordinates.y.min()
            if v.coordinates.x.min()<xmin:
                xmin = v.coordinates.x.min()

        # shift the coordinates for all volumes by the same amounts,
        # keeping them aligned
        for v in self.volumes:
            v.coordinates.z-=zmin
            v.coordinates.y-=ymin
            v.coordinates.x-=xmin

        # find the maximum depth
        max_n_depth = np.max([v.n_depth for v in self.volumes])

        # find the new max in each dimension
        ymax,zmax,xmax = -large_integer,-large_integer,-large_integer

        for v in self.volumes:
            if v.coordinates.z.max()>zmax:
                zmax = v.coordinates.z.max()
            if v.coordinates.y.max()>ymax:
                ymax = v.coordinates.y.max()
            if v.coordinates.x.max()>xmax:
                xmax = v.coordinates.x.max()

        # create accumulators
        sum_array = np.zeros((ymax+1,zmax+max_n_depth+1,xmax+1))
        counter_array = np.zeros((ymax+1,zmax+max_n_depth+1,xmax+1))
        
        for v in self.volumes:
            vol = v.get_volume()
            sy,sz,sx = vol.shape
            for y in range(sy):
                for x in range(sx):
                    ascan = vol[y,:,x]
                    ypos = v.coordinates.y[y,x]
                    xpos = v.coordinates.x[y,x]
                    zpos = v.coordinates.z[y,x]
                    sum_array[ypos,zpos:zpos+sz,xpos]+=self.summing_function(ascan)
                    counter_array[ypos,zpos:zpos+sz,xpos]+=1
                    
        sum_array[counter_array==0]=np.nan
        av = sum_array/counter_array

        if diagnostics:
            dB_clim = (40,80)
            plt.figure()
            plt.suptitle('full volume projections')
            plt.subplot(1,3,1)
            plt.imshow(20*np.log10(np.nanmean(av,0)),clim=dB_clim,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('x-z')
            plt.subplot(1,3,2)
            plt.imshow(20*np.log10(np.nanmean(av,1)),clim=dB_clim,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('x-y')
            plt.subplot(1,3,3)
            plt.imshow(20*np.log10(np.nanmean(av,2)).T,clim=dB_clim,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('y-z')

            plt.figure()
            plt.suptitle('central slices')
            plt.subplot(1,3,1)
            plt.imshow(20*np.log10(av[av.shape[0]//2,:,:]),clim=dB_clim,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('x-z')
            plt.subplot(1,3,2)
            plt.imshow(20*np.log10(av[:,av.shape[1]//2,:]),clim=dB_clim,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('x-y')
            plt.subplot(1,3,3)
            plt.imshow(20*np.log10(av[:,:,av.shape[2]//2].T),clim=dB_clim,aspect='auto',cmap='gray')
            plt.colorbar()
            plt.title('y-z')

            
            plt.figure()
            for k in range(av.shape[0]):
                plt.cla()
                plt.imshow(av[k,:,:])
                plt.pause(.1)
            plt.show()


class SyntheticVolume:

    def __init__(self,n_slow,n_depth,n_fast,diagnostics=False):
        self.dzf = 0.0
        self.dyf = 0.0
        self.dxf = 0.0
        
        self.dz = 0
        self.dy = 0
        self.dx = 0

        self.zstd = 0.06
        self.ystd = 0.08
        self.xstd = 0.1

        self.n_fast = n_fast
        self.n_slow = n_slow
        self.n_depth = n_depth
        
        self.yscanner = 0
        self.xscanner = 0
        
        cache_dir = '.synthetic_volume_cache'
        os.makedirs(cache_dir,exist_ok=True)
        rpower = 10000 # higher numbers = sparser objects 50000 creates just a few
        cache_fn = os.path.join(cache_dir,'%d_%d_%d_synthetic_source_%d.npy'%(n_slow,n_depth,n_fast,rpower))

        try:
            self.source = np.load(cache_fn)
        except FileNotFoundError:
            source_dims = (n_slow*2,n_depth*2,n_fast*2)

            self.source = np.random.random(source_dims)**rpower
            self.source[np.where(self.source<0.5)] = 0
            layer_thickness = 10
            for z in range(0,n_depth*2,layer_thickness*2):
                self.source[:,z:z+layer_thickness,:] = 0

            self.source = self.source*2000
            

            sphere_size = 11
            sphere = np.zeros((sphere_size,sphere_size,sphere_size))
            XX,YY,ZZ = np.meshgrid(np.arange(sphere_size),np.arange(sphere_size),np.arange(sphere_size))
            v = -1
            XX = XX-(sphere_size+v)/2.0
            YY = YY-(sphere_size+v)/2.0
            ZZ = ZZ-(sphere_size+v)/2.0
            rad = np.sqrt(XX**2+YY**2+ZZ**2)
            sphere[rad<=sphere_size/2-1] = 1

            self.source = spn.convolve(self.source,sphere).astype(np.complex128)

            noise = np.random.random(source_dims)*200
            self.source[self.source==0]=noise[self.source==0]
            
            np.save(cache_fn,self.source)
        
        if diagnostics:
            for k in range(self.source.shape[0]):
                plt.cla()
                plt.imshow(self.source[k,:,:])
                plt.pause(.00001)
        
        #self.history = [(self.dy,self.dz,self.dx)]
        self.history = []
        
    def step(self,volume_rigid=False):

        self.history.append((self.dy,self.dz,self.dx))
        
        self.dzf = self.dzf + np.random.randn()*self.zstd
        self.dyf = self.dyf + np.random.randn()*self.ystd
        self.dxf = self.dxf + np.random.randn()*self.xstd

        limit = 10
        
        if np.abs(self.dzf)>limit:
            self.dzf = 0.0
        if np.abs(self.dxf)>limit:
            self.dxf = 0.0
        if np.abs(self.dyf)>limit:
            self.dyf = 0.0

        #if not volume_rigid or (self.xscanner==(self.n_fast-1) and self.yscanner==(self.n_slow-1)):
        if not volume_rigid or (self.xscanner==0 and self.yscanner==0):
            self.dz = int(round(self.dzf))
            self.dy = int(round(self.dyf))
            self.dx = int(round(self.dxf))
            
        self.xscanner = (self.xscanner+1)%self.n_fast
        if self.xscanner==0:
            self.yscanner = (self.yscanner+1)%self.n_slow
        


    def get_bscan(self,diagnostics=False,volume_rigid=False):
        ascans = []
            
        for k in range(self.n_fast):
            self.step(volume_rigid)
            x = (self.xscanner-self.n_fast//2)+self.source.shape[2]//2+self.dx
            y = (self.yscanner-self.n_slow//2)+self.source.shape[0]//2+self.dy
            z1 = -self.n_depth//2+self.source.shape[1]//2+self.dz
            z2 = z1+self.n_depth
            ascans.append(self.source[y,z1:z2,x])
            
        bscan = np.array(ascans).T
        logging.info('xscanner: %d, yscanner: %d, dx: %d, dy: %d, dz: %d'%(self.xscanner,self.yscanner,self.dx,self.dy,self.dz))
        if diagnostics:
            plt.cla()
            plt.imshow(np.abs(bscan))
            plt.pause(.001)
        return bscan
        

    def plot_history(self):
        t = np.arange(len(self.history))
        y = [tup[0] for tup in self.history]
        z = [tup[1] for tup in self.history]
        x = [tup[2] for tup in self.history]
        plt.subplot(1,3,1)
        plt.plot(t,y)
        plt.xlabel('time')
        plt.ylabel('y')
        plt.subplot(1,3,2)
        plt.plot(t,z)
        plt.xlabel('time')
        plt.ylabel('z')
        plt.subplot(1,3,3)
        plt.plot(t,x)
        plt.xlabel('time')
        plt.ylabel('x')
        
    def save_volume(self,directory_name,diagnostics=False,volume_rigid=False):
        os.makedirs(directory_name,exist_ok=True)
        for k in range(self.n_slow):
            outfn = os.path.join(directory_name,'complex_bscan_stack_%05d.npy'%k)
            np.save(outfn,self.get_bscan(diagnostics,volume_rigid))
            logging.info('Saving B-scan %d to %s.'%(k,outfn))

    def save_volumes(self,directory_root,n,diagnostics=False,volume_rigid=False):
        for k in range(n):
            self.save_volume('%s_%03d'%(directory_root,k),diagnostics,volume_rigid)
            
