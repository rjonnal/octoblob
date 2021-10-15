import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from .ticktock import tick, tock
import scipy.ndimage as spn
import scipy.interpolate as spi
import imageio

################################# Intro ####################################
# This version of volume_tools is meant to be a drop-in replacement for the
# previous version, but with a completely different underlying methodology.
# The previous version was made faster by only registering sub-volumes, and
# thus necessitated lots of bookkeeping. This one is going to work by registering
# windowed volumes to a reference volume. It'll be slower, but simpler, with
# fewer corner cases to worry about, and fewer assumptions.
#############################################################################

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
large_integer = 10000000000

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

def rect_filter(shape,radii,diagnostics=False):
    f = np.zeros(shape)
    sy,sz,sx = shape
    wy,wz,wx = radii
    ZZ,YY,XX = np.meshgrid(np.arange(sz),np.arange(sy),np.arange(sx))
    ZZ = ZZ - sz/2.0
    YY = YY - sy/2.0
    XX = XX - sx/2.0

    zz = ZZ**2/(wz**2)
    yy = YY**2/(wy**2)
    xx = XX**2/(wx**2)

    rad = np.sqrt(zz+yy+xx)
    g = np.zeros(rad.shape)
    g[rad<=1] = 1

    if diagnostics:
        plt.figure()
        for k in range(sy):
            plt.clf()
            plt.imshow(g[k,:,:],clim=(g.min(),g.max()))
            plt.colorbar()
            plt.title('%s of %s'%(k+1,sy))
            plt.pause(.1)
        plt.close()

    g = np.fft.fftshift(g)
    return g


def show3d(vol,mode='center',aspect='auto'):
    sy,sz,sx = vol.shape
    temp = np.abs(vol)
    ncol,nrow = 3,1
    #plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
    if mode=='center':
        plt.subplot(1,3,1)
        plt.imshow(temp[sy//2,:,:],cmap='gray',aspect=aspect)
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(temp[:,sz//2,:],cmap='gray',aspect=aspect)
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(temp[:,:,sx//2].T,cmap='gray',aspect=aspect)
        plt.title('z-y')
    elif mode=='average':
        plt.subplot(1,3,1)
        plt.imshow(temp.mean(0),cmap='gray',aspect=aspect)
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(temp.mean(1),cmap='gray',aspect=aspect)
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(temp.mean(2).T,cmap='gray',aspect=aspect)
        plt.title('z-y')
    elif mode=='max':
        plt.subplot(1,3,1)
        plt.imshow(np.max(temp,axis=0),cmap='gray',aspect=aspect)
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(np.max(temp,axis=1),cmap='gray',aspect=aspect)
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(np.max(temp,axis=2).T,cmap='gray',aspect=aspect)
        plt.title('z-y')
    elif mode=='nxc':
        reg_coords = list(np.unravel_index(np.argmax(vol),vol.shape))
        plt.subplot(1,3,1)
        plt.imshow(temp[reg_coords[0],:,:],cmap='gray',aspect=aspect)
        plt.plot(reg_coords[2],reg_coords[1],'g+')
        plt.title('z-x')
        plt.subplot(1,3,2)
        plt.imshow(temp[:,reg_coords[1],:],cmap='gray',aspect=aspect)
        plt.plot(reg_coords[2],reg_coords[0],'g+')
        plt.title('y-x')
        plt.subplot(1,3,3)
        plt.imshow(temp[:,:,reg_coords[2]].T,cmap='gray',aspect=aspect)
        plt.plot(reg_coords[0],reg_coords[1],'g+')
        plt.title('z-y')
        

def nxc3d(ref,tar,diagnostics=False):

    # Differences from previous versions:
    # 1. We should expect not to receive NaN pixels
    # 2. Handle the upsampling/downsampling externally
    
    #ref = norm(ref)
    #tar = norm(tar)

    pref = np.zeros(ref.shape,dtype=ref.dtype)
    ptar = np.zeros(tar.shape,dtype=tar.dtype)

    pref[:] = norm(ref)
    ptar[:] = norm(tar)
    
    if diagnostics:
        show3d(pref)
        show3d(ptar)
        plt.show()
        
    n_slow,n_depth,n_fast = pref.shape

    #logging.info('Registering volumes of shape %dx%dx%d (slow x depth x fast).'%(n_slow,n_depth,n_fast))
    t0 = tick()

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
    nxc = np.real(np.fft.ifftn(fref*np.conj(ftar)))
    logging.info('Registration took %0.3f sec.'%dt)
    return nxc


class Coordinates:
    """A Coordinates object keeps track of the 3D coordinates for each A-scan in a Volume object."""
    def __init__(self,n_slow,n_depth,n_fast):
        self.x,self.y = np.meshgrid(np.arange(n_fast),np.arange(n_slow))
        self.z = np.zeros(self.x.shape,dtype=np.int)
        self.sy,self.sx = self.z.shape
        self.correlation = np.zeros(self.x.shape)
        
    def move_x(self,dx,boundaries):
        self.x[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]+=dx

    def move_y(self,dy,boundaries):
        self.y[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]+=dy

    def move_z(self,dz,boundaries):
        self.z[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]+=dz

    def set_correlation(self,corr,boundaries):
        self.correlation[boundaries.y1:boundaries.y2,boundaries.x1:boundaries.x2]=corr

        
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

    def __init__(self,bscan_folder,use_cache=True,diagnostics=False,hold_volume_in_ram=True,resampling=1):

        t0 = tick()
        
        self.bscan_folder = bscan_folder
        print(self.bscan_folder)
        self.bscan_filenames = sorted(glob.glob(os.path.join(self.bscan_folder,'*.npy')))
        self.resampling=resampling


        
        # determine volume shape from file list length and sample B-scan:
        self.n_slow = len(self.bscan_filenames)
        logging.info('Creating a Volume object based on %d bscans in %s.'%(self.n_slow,bscan_folder))
        
        temp = np.load(self.bscan_filenames[0])
        self.n_depth,self.n_fast = temp.shape


        
        # set default coordinates:
        self.coordinates = Coordinates(self.n_slow,self.n_depth,self.n_fast)

        self.moved = False
        
        self.use_cache = use_cache
        self.cache_dir = os.path.join(self.bscan_folder,'volume')
        self.cache_filename = os.path.join(self.cache_dir,'volume_%0.1f.npy'%self.resampling)
        
        os.makedirs(self.cache_dir,exist_ok=True)

        self.hold_volume_in_ram = hold_volume_in_ram

        volume = self.build_volume()
        self.unique_id = self.make_id(volume)
        logging.info('Initializing volume with id %s.'%self.unique_id)
        sy,sz,sx = volume.shape
        
        if self.hold_volume_in_ram:
            self.volume = volume

        self.ac_max_dict = {}
        self.is_reference = False

        self.y_grid = np.arange(0,sy)/self.resampling

    def make_id(self,volume):
        sy,sx,sz = volume.shape
        vals = []
        for y in range(0,sy,sy//3):
            for x in range(0,sx,sx//3):
                for z in range(0,sz,sz//3):
                    val = volume[y,x,z]
                    if np.isnan(val):
                        val = 0.0
                    vals.append(val)
        vals = tuple(vals)
        out = '%d'%hash(vals)
        out = out.replace('-','m')
        return out
    
    def build_volume(self,diagnostics=False):
        t0 = tick()

        if self.use_cache and os.path.exists(self.cache_filename):
            logging.info('Loading volume from %s.'%self.cache_filename)
            volume = np.load(self.cache_filename)
        else:
            logging.info('Building volume in %s.'%self.bscan_folder)
            temp = np.load(self.bscan_filenames[0])
            dtype = temp.dtype

            volume_temp = []
            for rf in self.bscan_filenames:
                temp = np.load(rf)
                is_stack = len(temp.shape)>2
                temp = np.abs(temp)
                if is_stack:
                    temp = np.nanmean(temp,axis=M_SCAN_DIMENSION)

                if diagnostics:
                    plt.cla()
                    plt.imshow(temp,cmap='gray')
                    plt.pause(.1)

                volume_temp.append(temp)

            volume_temp = np.array(volume_temp,dtype=dtype)

            #self.flythrough(0,volume=volume_temp)
            #self.flythrough(1,volume=volume_temp)
            #self.flythrough(2,volume=volume_temp)
            
            # resample volume
            if self.resampling==1:
                volume = volume_temp
            else:
                sy,sz,sx = volume_temp.shape
                
                ry_vec = np.arange(0,sy-1,1.0/self.resampling)
                rz_vec = np.arange(0,sz-1,1.0/self.resampling)
                rx_vec = np.arange(0,sx-1,1.0/self.resampling)

                ryy,rzz,rxx = np.meshgrid(ry_vec,rz_vec,rx_vec)

                y_vec = np.arange(sy)
                z_vec = np.arange(sz)
                x_vec = np.arange(sx)

                points = list(zip(ryy.ravel(),rzz.ravel(),rxx.ravel()))
                interpolator = spi.RegularGridInterpolator((y_vec,z_vec,x_vec),volume_temp)
                volume = interpolator(points)
                # volume = np.reshape(volume,(len(rx_vec),len(ry_vec),len(rz_vec)))
                # volume = np.reshape(volume,(len(rx_vec),len(rz_vec),len(ry_vec)))
                # volume = np.reshape(volume,(len(ry_vec),len(rx_vec),len(rz_vec)))
                # volume = np.reshape(volume,(len(ry_vec),len(rz_vec),len(rx_vec)))
                # volume = np.reshape(volume,(len(rz_vec),len(rx_vec),len(ry_vec)))
                volume = np.reshape(volume,(len(rz_vec),len(ry_vec),len(rx_vec)))
                volume = np.transpose(volume,(1,0,2))

            #self.flythrough(1,volume=volume)
            #self.flythrough(2,volume=volume)
            #sys.exit()
            np.save(self.cache_filename,volume)
            
        #self.flythrough(0,volume=volume)
        logging.info('Done; took %0.3f sec.'%tock(t0))
        
        return volume


    def write_tiffs(self,output_folder,filename_format='bscan_%05d.tif'):
        os.makedirs(output_folder,exist_ok=True)
        
        vol = self.get_volume()
        sy,sz,sx = vol.shape
        avol = np.abs(vol)
        vmax = np.nanmax(avol)
        vmin = np.nanmin(avol)

        avol = (avol - vmin)/(vmax-vmin)*(2**16-1)
        avol[np.isnan(avol)] = 0
        avol = np.round(avol).astype(np.uint16)
        for y in range(sy):
            outfn = os.path.join(output_folder,filename_format%y)
            imageio.imwrite(outfn,avol[y,:,:])
            print('Writing TIFF to %s.'%outfn)

        with open(os.path.join(output_folder,'raw_image_stats.txt'),'w') as fid:
            fid.write('volume max: %0.3f\n'%vmax)
            fid.write('volume min: %0.3f\n'%vmin)
        
    def get_volume(self,diagnostics=False):
        if self.hold_volume_in_ram:
            logging.info('get_volume returning volume in RAM.')
            return self.volume
        else:
            logging.info('get_volume returning result of build_volume().')
            return self.build_volume()

    def move(self,shifts,boundaries,nxc_max=0.0):
        if self.is_reference:
            try:
                assert not any(shifts)
            except AssertionError:
                logging.info('move: assertion error on reference')
                return
        self.coordinates.move_y(shifts[0],boundaries)
        self.coordinates.move_z(shifts[1],boundaries)
        self.coordinates.move_x(shifts[2],boundaries)
        self.coordinates.set_correlation(nxc_max,boundaries)
        self.moved = True


    def get_window(self,y,mode='gaussian',width=3):
        if mode=='gaussian':
            out = np.exp(-(self.y_grid-y)**2/(2*width**2))
        elif mode=='rect':
            out = np.zeros(self.y_grid.shape)
            out[np.where(np.abs(self.y_grid-y)<width)] = 1
            
        return out

    def flythrough(self,axis=1,volume=None):
        plt.figure()
        if volume is None:
            volume = self.get_volume()
        nframes = volume.shape[axis]
        for k in range(nframes):
            if axis==0:
                im = volume[k,:,:]
            elif axis==1:
                im = volume[:,k,:]
            elif axis==2:
                im = volume[:,:,k].T
            
            im = 20*np.log10(np.abs(im))
            plt.cla()
            plt.imshow(im,clim=(40,80),cmap='gray')
            plt.pause(.0001)
        plt.close()
        
    def register_to(self,reference_volume,sigma=10):

        rcache = '.registration_cache'
        os.makedirs(rcache,exist_ok=True)

        pair_id = reference_volume.unique_id+'_'+self.unique_id
        cache_fn = os.path.join(rcache,'reg_info_%s.npy'%pair_id)
        
        #self.flythrough(0)
        #self.flythrough(1)
        #sys.exit()
        
        t0 = tick()
        rvol = reference_volume.get_volume()
        tvol = self.get_volume()
        sy,sz,sx = tvol.shape
        cache_folder = '.register_to_cache'
        os.makedirs(cache_folder,exist_ok = True)

        try:
            volume_info = np.load(cache_fn)
        except:
            ac_max_key = '_'.join([str(s) for s in reference_volume.get_volume().shape])
            try:
                ac_max = self.ac_max_dict[ac_max_key]
            except Exception as e:
                nxc = nxc3d(rvol,rvol)
                ac_max = nxc.max()
                self.ac_max_dict[ac_max_key] = ac_max

            volume_info = []

            for y0 in np.arange(self.n_slow):
                temp = tvol.copy()
                temp = np.transpose(temp,(1,2,0))
                win = self.get_window(y0,'gaussian',sigma*self.resampling)
                xc_correction = float(len(win))/np.sum(win)
                temp = temp*win
                temp = np.transpose(temp,(2,0,1))

                #sy,sz,sx = temp.shape
                nxc = nxc3d(rvol,temp)

                #show3d(temp,mode='max')
                #plt.show()
                #plt.clf()
                #show3d(np.fft.fftshift(nxc),mode='max')
                #plt.pause(.1)
                #plt.show()
                #sys.exit()

                reg_coords = list(np.unravel_index(np.argmax(nxc),nxc.shape))
                nxc_max = np.max(nxc)/ac_max*xc_correction

                for idx in range(len(nxc.shape)):
                    if reg_coords[idx]>nxc.shape[idx]//2:
                        reg_coords[idx] = reg_coords[idx]-nxc.shape[idx]

                chunk_info = reg_coords+[nxc_max]
                volume_info.append(chunk_info)
                logging.info('Anchor %d of %d: %s'%(y0,self.n_slow,chunk_info))

            volume_info = np.array(volume_info)
            np.save(cache_fn,volume_info)
        
        for y0 in np.arange(self.n_slow):
            b = Boundaries(y0,y0+1,0,sz,0,sx)
            chunk_info = volume_info[y0]
            self.move(chunk_info[:3].astype(int),b,chunk_info[3])
            #plt.clf()
            #plt.imshow(self.coordinates.x-np.arange(sx))
            #plt.colorbar()
            #plt.pause(.0001)

        t1 = tock(t0)
        logging.info('register_to took %0.3f s'%t1)

        

class VolumeSeries:

    def __init__(self,reference_folder,resampling=1.0,sigma=10,signal_function=np.abs):
        self.volumes = []
        self.signal_function = signal_function
        self.resampling = resampling
        self.sigma = sigma
        self.add_reference(reference_folder)
        self.folder = os.path.join('registered','%s_%0.1f_%0.1f'%(reference_folder.strip('/').strip('\\'),self.resampling,self.sigma))
        os.makedirs(self.folder,exist_ok=True)


    def __getitem__(self,n):
        return self.volumes[n]

    def add_reference(self,volume_folder):
        vol = Volume(volume_folder,resampling=self.resampling)
        self.reference = vol

    def add_target(self,volume_folder):
        vol = Volume(volume_folder,resampling=self.resampling)
        self.volumes.append(vol)

    def register(self):
        info_folder = os.path.join(self.folder,'info')
        os.makedirs(info_folder,exist_ok=True)
        
        for v in self.volumes:
            v.register_to(self.reference,sigma=self.sigma)

    def render(self,threshold_percentile=0.0,diagnostics=False,display_function=lambda x: 20*np.log10(x),display_clim=None,make_bscan_flythrough=True,make_enface_flythrough=True):

        bscan_png_folder = os.path.join(self.folder,'bscans_png')
        enface_png_folder = os.path.join(self.folder,'enface')
        bscan_folder = os.path.join(self.folder,'bscans')
        diagnostics_folder = os.path.join(self.folder,'info')

        if make_bscan_flythrough:
            os.makedirs(bscan_png_folder,exist_ok=True)
        if make_enface_flythrough:
            os.makedirs(enface_png_folder,exist_ok=True)
            
        os.makedirs(bscan_folder,exist_ok=True)
        os.makedirs(diagnostics_folder,exist_ok=True)

        n_slow, n_depth, n_fast = self.volumes[0].get_volume().shape
        
        # find the maximum depth
        max_n_depth = np.max([v.n_depth for v in self.volumes])

        zmin = large_integer
        for v in self.volumes:
            #v.coordinates.z = -v.coordinates.z
            
            if v.coordinates.z.min()<zmin:
                zmin = v.coordinates.z.min()

        for v in self.volumes:
            v.coordinates.z = v.coordinates.z - zmin
        
        # find the new max in z
        zmax = -large_integer

        for v in self.volumes:
            if v.coordinates.z.max()>zmax:
                zmax = v.coordinates.z.max()

        sum_array = np.zeros((n_slow,zmax+max_n_depth,n_fast))
        counter_array = np.zeros((n_slow,zmax+max_n_depth,n_fast))

        y_slices = []
        x_slices = []
        z_slices = []
        
        for idx,v in enumerate(self.volumes):
            temp = np.zeros(sum_array.shape,dtype=np.complex128)
            vol = v.get_volume()
            sy,sz,sx = vol.shape

            # plt.figure()
            # plt.imshow(v.coordinates.z,interpolation='none')
            # plt.colorbar()
            # plt.title(idx)
            # plt.show()

            for y in range(sy):
                for x in range(sx):
                    ascan = vol[y,:,x]
                    ypos = v.coordinates.y[y,x]
                    xpos = v.coordinates.x[y,x]
                    zpos = v.coordinates.z[y,x]

                    if ypos>=0 and ypos<n_slow and xpos>=0 and xpos<n_fast:
                        temp[ypos,zpos:zpos+sz,xpos]+=self.signal_function(ascan)
                        counter_array[ypos,zpos:zpos+sz,xpos]+=1

            # np.save(os.path.join(info_folder,'xcoord_%05d.npy'%idx),v.coordinates.x)
            # np.save(os.path.join(info_folder,'ycoord_%05d.npy'%idx),v.coordinates.y)
            # np.save(os.path.join(info_folder,'zcoord_%05d.npy'%idx),v.coordinates.z)
            # np.save(os.path.join(info_folder,'corr_%05d.npy'%idx),v.coordinates.correlation)

            # with open(os.path.join(info_folder,'bscan_source_%05d.txt'%idx),'w') as fid:
            #     fid.write('%s\n'%v.bscan_folder)
            
            sum_array+=self.signal_function(temp)
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
                plt.suptitle('%s\nvolume %d'%(self.folder,idx))
                plt.subplot(1,3,1)
                plt.imshow(ys,cmap='gray',aspect='equal')
                plt.title('z-x')
                plt.subplot(1,3,2)
                plt.imshow(zs,cmap='gray',aspect='equal')
                plt.title('y-x')
                plt.subplot(1,3,3)
                plt.imshow(xs.T,cmap='gray',aspect='equal')
                plt.title('z-y')
                plt.savefig(os.path.join(diagnostics_folder,'single_volume_%05d_slices.png'%idx),dpi=150)

            plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
            plt.suptitle('%s\nfull volume projections'%self.folder)
            plt.subplot(1,3,1)
            plt.imshow(display_function(np.nanmean(av,0)),clim=display_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-x')
            plt.subplot(1,3,2)
            plt.imshow(display_function(np.nanmean(av,1)),clim=display_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('y-x')
            plt.subplot(1,3,3)
            plt.imshow(display_function(np.nanmean(av,2)).T,clim=display_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-y')
            plt.savefig(os.path.join(diagnostics_folder,'average_volume_projections.png'),dpi=150)

            
            plt.figure(figsize=(ncol*col_width_inches,nrow*row_height_inches),dpi=screen_dpi)
            plt.suptitle('%s\ncentral slices'%self.folder)
            plt.subplot(1,3,1)
            plt.imshow(display_function(av[av.shape[0]//2,:,:]),clim=display_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-x')
            plt.subplot(1,3,2)
            plt.imshow(display_function(av[:,av.shape[1]//2,:]),clim=display_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('y-x')
            plt.subplot(1,3,3)
            plt.imshow(display_function(av[:,:,av.shape[2]//2].T),clim=display_clim,aspect='equal',cmap='gray')
            plt.colorbar()
            plt.title('z-y')
            plt.savefig(os.path.join(diagnostics_folder,'average_volume_slices.png'),dpi=150)

        asy,asz,asx = av.shape
        save_dpi = 100.0
        fsz = asz/save_dpi
        fsx = asx/save_dpi
        plt.close('all')

        valid_values = av[~np.isnan(av)]
        valid_values = display_function(valid_values)

        if display_clim is None:
            display_clim = np.percentile(valid_values,(1,99.9))
        
        fsz = asz/save_dpi
        fsx = asx/save_dpi
        fig = plt.figure(figsize=(fsx,fsz),dpi=save_dpi*2)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xticks([])
        ax.set_yticks([])

        for k in range(asy):
            frame = av[k,:,:]
            np.save(os.path.join(bscan_folder,'bscan_%05d.npy'%k),frame)

        
        if make_bscan_flythrough:
            for k in range(asy):
                frame = av[k,:,:]
                frame[np.isnan(frame)] = display_clim[0]
                frame = display_function(frame)
                ax.clear()
                ax.imshow(frame,cmap='gray',interpolation='none',clim=display_clim)
                plt.savefig(os.path.join(bscan_png_folder,'bscan_%05d.png'%k),dpi=save_dpi)
                plt.pause(.000001)
            plt.close()

        fsy = asy/save_dpi
        fsx = asx/save_dpi
        fig = plt.figure(figsize=(fsx,fsy),dpi=save_dpi*2)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xticks([])
        ax.set_yticks([])
        
        if make_enface_flythrough:
            for k in range(asz):
                frame = av[:,k,:]
                frame[np.isnan(frame)] = display_clim[0]
                frame = display_function(frame)
                ax.clear()
                ax.imshow(frame,cmap='gray',interpolation='none',clim=display_clim)
                plt.savefig(os.path.join(enface_png_folder,'enface_%05d.png'%k),dpi=save_dpi)
                plt.pause(.000001)
            plt.close()

        plt.close('all')









        
class oldSyntheticVolume:

    def __init__(self,n_slow,n_depth,n_fast,diagnostics=False,sphere_diameter=11,motion=None,rpower=10000,regular=False,plane_thickness=0):
        # rpower: higher numbers = sparser objects 50000 creates just a few
        self.dzf = 0.0
        self.dyf = 0.0
        self.dxf = 0.0
        
        self.dz = 0
        self.dy = 0
        self.dx = 0

        self.zstd = 0.03
        self.ystd = 0.02
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
        
        cache_fn = os.path.join(cache_dir,'%d_%d_%d_synthetic_source_%d%s_%d_%d.npy'%(n_slow,n_depth,n_fast,rpower,regstring,sphere_diameter,plane_thickness))

        try:
            self.source = np.load(cache_fn)
        except FileNotFoundError:
            source_dims = (n_slow*2,n_depth*2,n_fast*2)

            self.source = np.random.random(source_dims)**rpower
            self.source[np.where(self.source<0.5)] = 0
            self.source[np.where(self.source)] = 1

            layer_thickness = 10
            for z in range(0,n_depth*2,layer_thickness*2):
                self.source[:,z:z+layer_thickness,:] = 0

            #sphere_diameter = 11
            sphere = np.zeros((sphere_diameter,sphere_diameter,sphere_diameter))
            XX,YY,ZZ = np.meshgrid(np.arange(sphere_diameter),np.arange(sphere_diameter),np.arange(sphere_diameter))
            v = -1
            XX = XX-(sphere_diameter+v)/2.0
            YY = YY-(sphere_diameter+v)/2.0
            ZZ = ZZ-(sphere_diameter+v)/2.0
            rad = np.sqrt(XX**2+YY**2+ZZ**2)
            sphere[rad<sphere_diameter/2-1] = 1

            self.source = spn.convolve(self.source,sphere)
            self.source = (self.source-np.min(self.source))/(np.max(self.source)-np.min(self.source))

            peak = 6000.0
            
            self.source = self.source*peak
            
            noise = np.random.standard_normal(source_dims)*np.sqrt(peak)+5*np.sqrt(peak)
            self.source = self.source + noise
            self.source[self.source<=1] = 1.0

            if plane_thickness:
                self.source[:,n_depth:n_depth+plane_thickness,:] = peak
            
            np.save(cache_fn,self.source)
        
        if diagnostics:
            for k in range(self.source.shape[0]):
                plt.cla()
                plt.imshow(np.abs(self.source[k,:,:]))
                plt.title(k)
                plt.pause(.00001)
            plt.close()
        
        #self.history = [(self.dy,self.dz,self.dx)]
        self.history = []
        self.scanner_history = []
        
    def step(self,volume_rigid=False,bscan_rigid=False,motion_factor=1.0,reference_rigid=False):

        self.history.append((self.dy,self.dz,self.dx))
        self.scanner_history.append(np.sqrt(self.yscanner**2+self.xscanner**2))

        if reference_rigid and len(self.history)<self.n_slow*self.n_fast:
            reference_rigid_factor = 0.0
        else:
            reference_rigid_factor = 1.0
        
        if self.motion is None:
            self.dzf = self.dzf + np.random.randn()*self.zstd*motion_factor*reference_rigid_factor
            self.dyf = self.dyf + np.random.randn()*self.ystd*motion_factor*reference_rigid_factor
            self.dxf = self.dxf + np.random.randn()*self.xstd*motion_factor*reference_rigid_factor

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

        make_move = False
        if not bscan_rigid and not volume_rigid:
            make_move = True
        elif not bscan_rigid and volume_rigid:
            sys.exit('bscan_rigid is False but volume_rigid is True--inconsistent.')
        elif bscan_rigid and not volume_rigid:
            make_move = self.xscanner==0
        elif bscan_rigid and volume_rigid:
            make_move = (self.xscanner==0 and self.yscanner==0)
        else:
            sys.exit('something bad has happened.')
        
        if make_move:
            self.dz = int(round(self.dzf))
            self.dy = int(round(self.dyf))
            self.dx = int(round(self.dxf))
            
        self.xscanner = (self.xscanner+1)%self.n_fast
        if self.xscanner==0:
            self.yscanner = (self.yscanner+1)%self.n_slow
        


    def get_bscan(self,diagnostics=False,volume_rigid=False,bscan_rigid=False,motion_factor=1.0,reference_rigid=False):
        ascans = []
            
        for k in range(self.n_fast):
            self.step(volume_rigid=volume_rigid,bscan_rigid=bscan_rigid,motion_factor=motion_factor,reference_rigid=reference_rigid)
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


        
    def save_volume(self,folder_name,diagnostics=False,volume_rigid=False,bscan_rigid=False,motion_factor=1.0,reference_rigid=False):
        os.makedirs(folder_name,exist_ok=True)
        for k in range(self.n_slow):
            outfn = os.path.join(folder_name,'complex_bscan_stack_%05d.npy'%k)
            np.save(outfn,self.get_bscan(diagnostics,volume_rigid=volume_rigid,
                                         bscan_rigid=bscan_rigid,motion_factor=motion_factor,
                                         reference_rigid=reference_rigid))
            logging.info('Saving B-scan %d to %s.'%(k,outfn))

    def save_volumes(self,folder_root,n,diagnostics=False,volume_rigid=False,bscan_rigid=False,motion_factor=1.0,reference_rigid=False):
        for k in range(n):
            self.save_volume('%s_%03d'%(folder_root,k),diagnostics,volume_rigid=volume_rigid,bscan_rigid=bscan_rigid,motion_factor=motion_factor,reference_rigid=reference_rigid)
        info_folder = os.path.join(os.path.split(folder_root)[0],'info')
        os.makedirs(info_folder,exist_ok=True)
        self.plot_history()
        plt.savefig(os.path.join(info_folder,'eye_movements.png'),dpi=300)
        np.save(os.path.join(info_folder,'eye_movements.npy'),np.array(self.history))

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
            
