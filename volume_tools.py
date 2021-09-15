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
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# for loading M-scan series as volumes, we average the abs along this dimension:
M_SCAN_DIMENSION = 2
screen_dpi = 100
col_width_inches = 2.5
row_height_inches = 2.5

class Space:

    def __init__(self,initialization_volume):
        self.n_slow,self.n_depth,self.n_fast = initialization_volume.shape
        logging.info('Initializing coordinate space with shape %d x %d x %d (slow x depth x fast).'%(self.n_slow,self.n_depth,self.n_fast))
        self.entries = []
        self.entries.append((initialization_volume,(0,0,0)))


    def put(self,volume,coordinates):
        pass

def norm(im):
    return (im - np.nanmean(im)/np.nanstd(im))


def gaussian_filter(shape,sigmas,diagnostics=False):
    f = np.zeros(shape)
    sy,sz,sx = shape
    wy,wz,wx = sigmas
    ZZ,YY,XX = np.meshgrid(np.arange(sz),np.arange(sy),np.arange(sx))
    ZZ = ZZ - sz/2.0
    YY = YY - sy/2.0
    XX = XX - sx/2.0

    zz = ZZ**2/(2*wz**2)
    yy = YY**2/(2*wy**2)
    xx = XX**2/(2*wx**2)
    g = np.exp(-(xx+yy+zz))

    if diagnostics:
        plt.figure()
        for k in range(sy):
            plt.clf()
            plt.imshow(g[k,:,:],clim=(g.min(),g.max()))
            plt.colorbar()
            plt.pause(.5)
        plt.close()

    g = np.fft.fftshift(g)
    return g

def show3d(vol,mode='center'):
    sy,sz,sx = vol.shape
    temp = np.abs(vol)
    ncol,nrow = 3,1
    plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
    if mode=='center':
        plt.subplot(1,3,1)
        plt.imshow(temp[sy//2,:,:],cmap='gray',aspect='auto')
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(temp[:,sz//2,:],cmap='gray',aspect='auto')
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(temp[:,:,sx//2].T,cmap='gray',aspect='auto')
        plt.title('z-y')
    elif mode=='average':
        plt.subplot(1,3,1)
        plt.imshow(temp.mean(0),cmap='gray',aspect='auto')
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(temp.mean(1),cmap='gray',aspect='auto')
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(temp.mean(2).T,cmap='gray',aspect='auto')
        plt.title('z-y')
    elif mode=='nxc':
        reg_coords = list(np.unravel_index(np.argmax(vol),vol.shape))
        plt.subplot(1,3,1)
        plt.imshow(temp[reg_coords[0],:,:],cmap='gray',aspect='auto')
        plt.plot(reg_coords[2],reg_coords[1],'g+')
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(temp[:,reg_coords[1],:],cmap='gray',aspect='auto')
        plt.plot(reg_coords[2],reg_coords[0],'g+')
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(temp[:,:,reg_coords[2]].T,cmap='gray',aspect='auto')
        plt.plot(reg_coords[0],reg_coords[1],'g+')
        plt.title('z-y')
        
        
        
        

def nxc3d(ref,tar,downsample=1,diagnostics=False,border_size=0):
    #ref = norm(ref)
    #tar = norm(tar)

    pref = np.zeros(ref.shape,dtype=ref.dtype)
    ptar = np.zeros(tar.shape,dtype=tar.dtype)

    if border_size:
        pref[border_size:-border_size,border_size:-border_size,border_size:-border_size] = ref[border_size:-border_size,border_size:-border_size,border_size:-border_size]
        ptar[border_size:-border_size,border_size:-border_size,border_size:-border_size] = tar[border_size:-border_size,border_size:-border_size,border_size:-border_size]
    else:
        pref = ref
        ptar = tar

    if diagnostics:
        show3d(pref)
        show3d(ptar)
        plt.show()
        
    # default behavior for NaN values:
    pref[np.isnan(pref)] = 0.0
    ptar[np.isnan(ptar)] = 0.0
        
    n_slow,n_depth,n_fast = pref.shape

    logging.info('Registering volumes of shape %dx%dx%d (slow x depth x fast).'%(n_slow,n_depth,n_fast))
    t0 = tick()
    
    # if diagnostics:
    #     plt.figure()
    #     plt.subplot(4,2,1)
    #     plt.imshow(np.mean(np.abs(pref),axis=1))
    #     plt.subplot(4,2,2)
    #     plt.imshow(np.mean(np.abs(ptar),axis=1))
    #     plt.subplot(4,2,5)
    #     plt.imshow(np.mean(np.abs(pref),axis=0))
    #     plt.subplot(4,2,6)
    #     plt.imshow(np.mean(np.abs(ptar),axis=0))
        
    pref = pref[::downsample,::downsample,::downsample]
    ptar = ptar[::downsample,::downsample,::downsample]

    # if diagnostics:
    #     plt.subplot(4,2,3)
    #     plt.imshow(np.mean(np.abs(pref),axis=1))
    #     plt.subplot(4,2,4)
    #     plt.imshow(np.mean(np.abs(ptar),axis=1))
    #     plt.subplot(4,2,7)
    #     plt.imshow(np.mean(np.abs(pref),axis=0))
    #     plt.subplot(4,2,8)
    #     plt.imshow(np.mean(np.abs(ptar),axis=0))

    
    rsx,rsz,rsy = pref.shape
    tsx,tsz,tsy = ptar.shape
    
    sx = max(rsx,tsx)
    sz = max(rsz,tsz)
    sy = max(rsy,tsy)
    
    t0 = tick()
    s = (sx,sz,sy)
    fref = np.fft.fftn(pref,s=s)
    ftar = np.fft.fftn(ptar,s=s)
    dt = tock(t0)
    logging.info('Registration took %0.3f sec with downsampling of %d.'%(dt,downsample))
    
    return np.real(np.fft.ifftn(fref*np.conj(ftar)))


class Coordinates:
    """A Coordinates object keeps track of the 3D coordinates for each A-scan in a Volume object."""
    def __init__(self,n_slow,n_depth,n_fast):
        self.x,self.y = np.meshgrid(np.arange(n_fast),np.arange(n_slow))
        self.z = np.zeros(self.x.shape,dtype=np.int)
        self.sy,self.sx = self.z.shape
        
    def move_x(self,dx,boundaries):
        self.x[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]+=dx

    def move_y(self,dy,boundaries):
        self.y[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]+=dy

    def move_z(self,dz,boundaries):
        self.z[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]+=dz


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

    def __init__(self,bscan_directory,cache=True,diagnostics=False,hold_volume_in_ram=True):
        t0 = tick()
        
        self.bscan_directory = bscan_directory
        self.bscan_filenames = sorted(glob.glob(os.path.join(self.bscan_directory,'*.npy')))

        # determine volume shape from file list length and sample B-scan:
        self.n_slow = len(self.bscan_filenames)
        temp = np.load(self.bscan_filenames[0])
        self.n_depth,self.n_fast = temp.shape

        # set default coordinates:
        self.coordinates = Coordinates(self.n_slow,self.n_depth,self.n_fast)

        self.moved = False
        
        self.cache = cache
        self.cache_dir = os.path.join(self.bscan_directory,'volume')
        os.makedirs(self.cache_dir,exist_ok=True)

        self.hold_volume_in_ram = hold_volume_in_ram
        if self.hold_volume_in_ram:
            self.volume = self.build_volume()

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
        if self.hold_volume_in_ram:
            logging.info('get_volume returning volume in RAM.')
            return self.volume
        else:
            if self.cache:
                vol_fn = os.path.join(self.cache_dir,'volume.npy')

                if os.path.exists(vol_fn):
                    t0 = tick()
                    out_vol = np.load(vol_fn)
                    logging.info('Using cached version at %s; took %0.3f sec.'%(vol_fn,tock(t0)))
                    return out_vol

                else:
                    out_vol = self.build_volume(diagnostics=diagnostics)
                    np.save(vol_fn,out_vol)
                    return out_vol
            else:
                out_vol = self.build_volume(diagnostics=diagnostics)
                return out_vol

    def move(self,shifts,boundaries):
        self.coordinates.move_y(shifts[0],boundaries)
        self.coordinates.move_z(shifts[1],boundaries)
        self.coordinates.move_x(shifts[2],boundaries)
        self.moved = True


    def get_block(self,b,volume=None,diagnostics=False):
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

        t0 = tick()
        xc = np.copy(self.coordinates.x)
        yc = np.copy(self.coordinates.y)
        zc = np.copy(self.coordinates.z)

        if volume is None:
            volume = self.get_volume()

        # quickly check if this is a reference volume for quicker block assembly:
        if not self.moved:
            block = volume[b.y1:b.y2,b.z1:b.z2,b.x1:b.x2]
        else:
            block = np.ones(b.shape,dtype=volume.dtype)*np.nan
            # Ugh this is going to be so slow. I'm too stupid to vectorize it.
            for yput in range(b.y1,b.y2):
                for xput in range(b.x1,b.x2):
                    yget,xget = np.where((xc==xput)*(yc==yput))
                    if len(yget):
                        yget = yget[0]
                        xget = xget[0]
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
                        block[yput-b.y1,zget1:zget2,xput-b.x1] = volume[yget,zput1:zput2,xget]
                        
        if diagnostics:
            ncol,nrow=1,1
            plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
            plt.imshow(np.abs(block[b.shape[0]//2,:,:]))

        t1 = tock(t0)
        logging.info('get_block took %0.3f s'%t1)

        return block

    def register_to(self,reference_volume,boundaries,limits=(np.inf,np.inf,np.inf),downsample=1,diagnostics=False,border_size=0,nxc_filter=None):
        
        t0 = tick()
        rvol = reference_volume.get_block(boundaries,diagnostics=diagnostics)
        tvol = self.get_block(boundaries,diagnostics=diagnostics)


        nxc = nxc3d(rvol,tvol,downsample=downsample,diagnostics=diagnostics,border_size=border_size)
        if nxc_filter is not None:
            nxc = nxc*nxc_filter
        
        reg_coords = list(np.unravel_index(np.argmax(nxc),nxc.shape))
        for idx in range(len(nxc.shape)):
            if reg_coords[idx]>nxc.shape[idx]//2:
                reg_coords[idx] = reg_coords[idx]-nxc.shape[idx]


                
        upsampled_reg_coords = [rc*downsample for rc in reg_coords]

        limited_coords = []
        for c,limit in zip(upsampled_reg_coords,limits):
            if np.abs(c)>limit:
                limited_coords.append(0)
            else:
                limited_coords.append(c)
        
        self.move(limited_coords,boundaries)
        t1 = tock(t0)
        logging.info('register_to took %0.3f s; shifts were %d, %d, %d; nxc max %0.1f'%tuple([t1]+limited_coords+[np.max(nxc)]))

        
    def register_to0(self,reference_volume,boundaries,downsample=1,diagnostics=False,border_size=0):
        
        nxc = nxc3d(reference_volume.get_volume(),self.get_volume(),downsample=downsample,diagnostics=diagnostics,border_size=border_size)
        reg_coords = list(np.unravel_index(np.argmax(nxc),nxc.shape))
        for idx in range(len(nxc.shape)):
            if reg_coords[idx]>nxc.shape[idx]//2:
                reg_coords[idx] = reg_coords[idx]-nxc.shape[idx]

        if diagnostics:
            show3d(nxc,'nxc')
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


    def render(self,output_directory,diagnostics=False):

        os.makedirs(output_directory,exist_ok=True)
        
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
        #sum_array = np.zeros((ymax+1,zmax+max_n_depth+1,xmax+1))
        #counter_array = np.zeros((ymax+1,zmax+max_n_depth+1,xmax+1))

        # something a little bit funky about initializing these: why
        # is z different?
        sum_array = np.zeros((ymax+1,zmax+max_n_depth,xmax+1))
        counter_array = np.zeros((ymax+1,zmax+max_n_depth,xmax+1))

        y_slices = []
        x_slices = []
        z_slices = []
        
        for idx,v in enumerate(self.volumes):
            temp = np.zeros(sum_array.shape,dtype=np.complex128)
            vol = v.get_volume()
            sy,sz,sx = vol.shape
            for y in range(sy):
                for x in range(sx):
                    ascan = vol[y,:,x]
                    ypos = v.coordinates.y[y,x]
                    xpos = v.coordinates.x[y,x]
                    zpos = v.coordinates.z[y,x]
                    temp[ypos,zpos:zpos+sz,xpos]+=ascan
                    counter_array[ypos,zpos:zpos+sz,xpos]+=1

            np.save(os.path.join(output_directory,'volume_%05d.npy'%idx),temp)
            
            sum_array+=self.summing_function(temp)
            # store some slices of temp for debugging:
            temp = np.abs(temp)
            
            y_slices.append(temp[temp.shape[0]//2,:,:])
            x_slices.append(temp[:,:,temp.shape[2]//2])
            z_slices.append(temp[:,temp.shape[1]//2,:])
                    
        sum_array[counter_array==0]=np.nan
        av = sum_array/counter_array

        if diagnostics:
            dB_clim = None#(40,80)
            ncol,nrow = 3,1
            for idx,(ys,zs,xs) in enumerate(zip(y_slices,z_slices,x_slices)):
                plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
                plt.suptitle('%s\nvolume %d'%(output_directory,idx))
                plt.subplot(1,3,1)
                plt.imshow(ys,cmap='gray',aspect='equal')
                plt.title('z-x')
                plt.subplot(1,3,2)
                plt.imshow(zs,cmap='gray',aspect='equal')
                plt.title('y-x')
                plt.subplot(1,3,3)
                plt.imshow(xs.T,cmap='gray',aspect='equal')
                plt.title('z-y')
                plt.savefig(os.path.join(output_directory,'single_volume_%05d_slices.png'%idx),dpi=150)

            dispfunc = lambda x: x
                

            plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
            plt.suptitle('%s\nfull volume projections'%output_directory)
            plt.subplot(1,3,1)
            plt.imshow(dispfunc(np.nanmean(av,0)),clim=dB_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-x')
            plt.subplot(1,3,2)
            plt.imshow(dispfunc(np.nanmean(av,1)),clim=dB_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('y-x')
            plt.subplot(1,3,3)
            plt.imshow(dispfunc(np.nanmean(av,2)).T,clim=dB_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-y')
            plt.savefig(os.path.join(output_directory,'average_volume_projections.png'),dpi=150)

            plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
            plt.suptitle('%s\ncentral slices'%output_directory)
            plt.subplot(1,3,1)
            plt.imshow(dispfunc(av[av.shape[0]//2,:,:]),clim=dB_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-x')
            plt.subplot(1,3,2)
            plt.imshow(dispfunc(av[:,av.shape[1]//2,:]),clim=dB_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('y-x')
            plt.subplot(1,3,3)
            plt.imshow(dispfunc(av[:,:,av.shape[2]//2].T),clim=dB_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-y')
            plt.savefig(os.path.join(output_directory,'average_volume_slices.png'),dpi=150)

            if False:

                flythrough_directory = os.path.join(output_directory,'flythrough')
                os.makedirs(flythrough_directory,exist_ok=True)

                plt.figure()
                for k in range(av.shape[0]):
                    plt.cla()
                    plt.imshow(av[k,:,:])
                    plt.savefig(os.path.join(flythrough_directory,'bscan_%03d.png'%k),dpi=150)
                    #plt.pause(.1)

            plt.close('all')


class SyntheticVolume:

    def __init__(self,n_slow,n_depth,n_fast,diagnostics=False,sphere_diameter=11,motion=None,rpower=10000,regular=False):
        # rpower: higher numbers = sparser objects 50000 creates just a few
        self.dzf = 0.0
        self.dyf = 0.0
        self.dxf = 0.0
        
        self.dz = 0
        self.dy = 0
        self.dx = 0

        self.zstd = 0.03
        self.ystd = 0.03
        self.xstd = 0.03

        self.motion = motion
        
        self.n_fast = n_fast
        self.n_slow = n_slow
        self.n_depth = n_depth
        
        self.yscanner = 0
        self.xscanner = 0
        
        cache_dir = '.synthetic_volume_cache'
        os.makedirs(cache_dir,exist_ok=True)

        if regular:
            regstring = '_reg'
        else:
            regstring = '_rand'
        
        cache_fn = os.path.join(cache_dir,'%d_%d_%d_synthetic_source_%d%s.npy'%(n_slow,n_depth,n_fast,rpower,regstring))

        try:
            self.source = np.load(cache_fn)
        except FileNotFoundError:
            source_dims = (n_slow*2,n_depth*2,n_fast*2)

            if not regular:
                self.source = np.random.random(source_dims)**rpower
                self.source[np.where(self.source<0.5)] = 0
            else:
                self.source = np.zeros(source_dims)
                for y in range(0+np.random.randint(sphere_diameter*3),n_slow*2,sphere_diameter*3):
                    for x in range(0+np.random.randint(sphere_diameter*3),n_fast*2,sphere_diameter*3):
                        for z in range(0+np.random.randint(sphere_diameter*3),n_depth*2,sphere_diameter*3):
                            self.source[y,z,x] = 1.0
                
            layer_thickness = 10
            for z in range(0,n_depth*2,layer_thickness*2):
                self.source[:,z:z+layer_thickness,:] = 0

            self.source = self.source*2000
            
            #sphere_diameter = 11
            sphere = np.zeros((sphere_diameter,sphere_diameter,sphere_diameter))
            XX,YY,ZZ = np.meshgrid(np.arange(sphere_diameter),np.arange(sphere_diameter),np.arange(sphere_diameter))
            v = -1
            XX = XX-(sphere_diameter+v)/2.0
            YY = YY-(sphere_diameter+v)/2.0
            ZZ = ZZ-(sphere_diameter+v)/2.0
            rad = np.sqrt(XX**2+YY**2+ZZ**2)
            sphere[rad<sphere_diameter/2-1] = 1

            self.source = spn.convolve(self.source,sphere).astype(np.complex128)

            noise = np.random.random(source_dims)*200
            self.source[self.source==0]=noise[self.source==0]
            
            np.save(cache_fn,self.source)
        
        if diagnostics:
            for k in range(self.source.shape[0]):
                plt.cla()
                plt.imshow(self.source[k,:,:])
                plt.pause(1)
            sys.exit()
                
        
        #self.history = [(self.dy,self.dz,self.dx)]
        self.history = []
        self.scanner_history = []
        
    def step(self,volume_rigid=False,motion_factor=1.0):

        self.history.append((self.dy,self.dz,self.dx))
        self.scanner_history.append(np.sqrt(self.yscanner**2+self.xscanner**2))
        
        if self.motion is None:
            self.dzf = self.dzf + np.random.randn()*self.zstd*motion_factor
            self.dyf = self.dyf + np.random.randn()*self.ystd*motion_factor
            self.dxf = self.dxf + np.random.randn()*self.xstd*motion_factor

            limit = 10

            if np.abs(self.dzf)>limit:
                self.dzf = 0.0
            if np.abs(self.dxf)>limit:
                self.dxf = 0.0
            if np.abs(self.dyf)>limit:
                self.dyf = 0.0

        else:
            self.dzf = self.dzf + self.motion[1]
            self.dyf = self.dyf + self.motion[0]
            self.dxf = self.dxf + self.motion[2]

        #if not volume_rigid or (self.xscanner==(self.n_fast-1) and self.yscanner==(self.n_slow-1)):
        if not volume_rigid or (self.xscanner==0 and self.yscanner==0):
            self.dz = int(round(self.dzf))
            self.dy = int(round(self.dyf))
            self.dx = int(round(self.dxf))
            
        self.xscanner = (self.xscanner+1)%self.n_fast
        if self.xscanner==0:
            self.yscanner = (self.yscanner+1)%self.n_slow
        


    def get_bscan(self,diagnostics=False,volume_rigid=False,motion_factor=1.0):
        ascans = []
            
        for k in range(self.n_fast):
            self.step(volume_rigid,motion_factor=motion_factor)
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
        scanner_zeros = np.where(np.array(self.scanner_history)==0)[0]
        
        plt.figure(figsize=(3*col_width_inches,row_height_inches),dpi=screen_dpi)
        plt.subplot(1,3,1)
        plt.plot(t,y)
        for scanner_zero in scanner_zeros:
            plt.axvline(scanner_zero,color='r')
        plt.xlabel('time')
        plt.ylabel('y')
        plt.subplot(1,3,2)
        plt.plot(t,z)
        for scanner_zero in scanner_zeros:
            plt.axvline(scanner_zero,color='r')
        plt.xlabel('time')
        plt.ylabel('z')
        plt.subplot(1,3,3)
        plt.plot(t,x)
        for scanner_zero in scanner_zeros:
            plt.axvline(scanner_zero,color='r')
        plt.xlabel('time')
        plt.ylabel('x')


        
    def save_volume(self,directory_name,diagnostics=False,volume_rigid=False,motion_factor=1.0):
        os.makedirs(directory_name,exist_ok=True)
        for k in range(self.n_slow):
            outfn = os.path.join(directory_name,'complex_bscan_stack_%05d.npy'%k)
            np.save(outfn,self.get_bscan(diagnostics,volume_rigid,motion_factor=motion_factor))
            logging.info('Saving B-scan %d to %s.'%(k,outfn))

    def save_volumes(self,directory_root,n,diagnostics=False,volume_rigid=False,motion_factor=1.0):
        for k in range(n):
            self.save_volume('%s_%03d'%(directory_root,k),diagnostics,volume_rigid,motion_factor=motion_factor)
        info_directory = os.path.join(os.path.split(directory_root)[0],'info')
        os.makedirs(info_directory,exist_ok=True)
        self.plot_history()
        plt.savefig(os.path.join(info_directory,'eye_movements.png'),dpi=300)
        np.save(os.path.join(info_directory,'eye_movements.npy'),np.array(self.history))

def make_simple_volume_series(folder_name):
    sx = 7
    sy = 6
    sz = 5

    x,y,z = 2,1,1
    width = 1

    n_vol = 3

    src = np.random.random((sy,sz,sx))*100

    for v in range(n_vol):
        out = np.copy(src)
        out[y+v:y+v+width,z+v:z+v+width,x+v:x+v+width] = 1000
        out_folder = os.path.join(folder_name,'synthetic_%02d'%v)
        os.makedirs(out_folder,exist_ok=True)
        for ny in range(sy):
            bscan = out[ny,:,:].astype(np.complex128)
            plt.cla()
            plt.imshow(np.abs(bscan),clim=(0,5000))
            plt.title('%d, %d'%(v,ny))
            plt.pause(.1)
            outfn = os.path.join(out_folder,'complex_bscan_stack_%05d.npy'%ny)
            np.save(outfn,bscan)
            
