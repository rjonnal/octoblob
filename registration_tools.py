import glob,sys,os
import numpy as np
from matplotlib import pyplot as plt

def rigid_register(ref,tar,max_shift=None,diagnostics=False,ref_pre_fft=False):
    # register two complex b-scans as rigid bodies

    # allows one-time fft2 of ref, for register_series below
    if not ref_pre_fft:
        fref = np.fft.fft2(ref)
    else:
        fref = ref
        
    ftar = np.conj(np.fft.fft2(tar))
    fprod = ftar*fref
    xc = np.abs(np.fft.ifft2(fprod))

    if diagnostics:
        plt.subplot(1,2,1)
        plt.cla()
        # use fftshift to put the peak in the middle for easier inspection
        plt.imshow(np.fft.fftshift(xc),interpolation='none')
        plt.title('cross corr fftshifted in image')

    
    sy,sx = ref.shape
    if not max_shift is None:
        XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))
        XX = XX-sx//2
        YY = YY-sy//2
        rad = np.sqrt(XX**2+YY**2)
        mask = np.zeros(rad.shape)
        mask[np.where(rad<=max_shift)]=1.0
        mask = np.fft.fftshift(mask)
        xc = xc * mask
        
    peaky,peakx = np.unravel_index(np.argmax(xc),xc.shape)
    xc_peak = xc[peaky,peakx]
    
    if peaky>xc.shape[0]//2:
        peaky=peaky-xc.shape[0]
    if peakx>xc.shape[1]//2:
        peakx=peakx-xc.shape[1]

    if diagnostics:
        plt.subplot(1,2,2)
        plt.cla()
        plt.imshow(np.fft.fftshift(xc),interpolation='none',aspect='auto')
        disp_x,disp_y = peakx+sx//2,peaky+sy//2
        plt.plot(disp_x,disp_y,'rx',alpha=0.2)
        plt.xlim((disp_x-15,disp_x+15))
        plt.ylim((disp_y-15,disp_y+15))
        plt.title('masked cross corr peak at (%d,%d)\nfftshifted in image'%(peaky,peakx))
        plt.pause(.001)
        
    return peakx,peaky,xc


def rigid_shift(ref,tar,max_shift=None,diagnostics=False,ref_pre_fft=False):
    # use rigid_register above and return a correctly shifted version of target (tar)
    
    dx,dy,xc = rigid_register(ref,tar,max_shift,diagnostics,ref_pre_fft)
    #new_tar = np.ones(tar.shape,dtype=tar.dtype)*np.min(tar)
    new_tar = np.zeros(tar.shape,dtype=tar.dtype)
    
    print(dx,dy)
    
    if dx>0:
        put_x1 = dx
        put_x2 = tar.shape[1]
        get_x1 = 0
        get_x2 = put_x2-put_x1
        
    if dy>0:
        put_y1 = dy
        put_y2 = tar.shape[0]
        get_y1 = 0
        get_y2 = put_y2-put_y1

    if dx<=0:
        get_x1 = -dx
        get_x2 = tar.shape[1]
        put_x1 = 0
        put_x2 = get_x2-get_x1
        
    if dy<=0:
        get_y1 = -dy
        get_y2 = tar.shape[0]
        put_y1 = 0
        put_y2 = get_y2-get_y1

    
    new_tar[put_y1:put_y2,put_x1:put_x2] = tar[get_y1:get_y2,get_x1:get_x2]
    
    if diagnostics:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.abs(tar),aspect='auto')
        plt.subplot(1,2,2)
        plt.imshow(np.abs(new_tar),aspect='auto')
        plt.title('rigid shift')
        
    return new_tar
    

def register_series(ref_fn,fn_list,output_directory=None,max_shift=None,diagnostics=False,overwrite=False):

    if output_directory is None:
        data_dir,basename = os.path.split(ref_fn)
        tag = os.path.splitext(basename)[0]
        output_directory = os.path.join(data_dir,'%s_registered'%tag)

    rinfo_directory = os.path.join(output_directory,'registration_info')

    try:
        os.makedirs(output_directory,exist_ok=overwrite)
        os.makedirs(rinfo_directory,exist_ok=overwrite)
    except FileExistsError as fee:
        print('%s exists. Please delete or specify a different output_directory.'%output_directory)


    
    try:
        median_autocorrelation = np.loadtxt(os.path.join(rinfo_directory,'median_autocorrelation.txt'))[0]
    except:
        # estimate ideal cross-correlation and build stack:
        xc_vec = []
        for fn in fn_list:
            f = np.load(fn)
            dx,dy,xc = rigid_register(f,f)
            xc_vec.append(xc.max())
        median_autocorrelation = np.median(xc_vec)
        np.savetxt(os.path.join(rinfo_directory,'median_autocorrelation.txt'),[np.median(xc_vec)])


    try:
        xc_stack = np.load(os.path.join(rinfo_directory,'cross_correlation_stack.npy'))
    except:
        xc_stack = []
        fref = np.fft.fft2(np.load(ref_fn))
        
        for fn in fn_list:
            f = np.load(fn)
            dx,dy,xc = rigid_register(f,fref,ref_pre_fft=True)
            xc_stack.append(xc)
            
        xc_stack = np.array(xc_stack)
        np.save(os.path.join(rinfo_directory,'cross_correlation_stack.npy'),xc_stack)

    print('median autocorrelation = %0.3f'%median_autocorrelation)
    sys.exit()
    # the first step is to use pairwise shifts to create confidence limits for the later, global registration:
    try:
        cdx_vec = np.loadtxt(os.path.join(rinfo_directory,'cdx.txt')).astype(np.int16)
        cdy_vec = np.loadtxt(os.path.join(rinfo_directory,'cdy.txt')).astype(np.int16)
        ref_idx = int(np.loadtxt(os.path.join(rinfo_directory,'ref_idx.txt'))[0])
    except:
        cdx_vec = []
        cdy_vec = []



        

        
        print(np.median(xc_vec))
        plt.plot(xc_vec)
        plt.show()
        
        for idx,(fn1,fn2) in enumerate(zip(fn_list[:-1],fn_list[1:])):

            # ref_idx is the index of the item in cdx_vec,cdy_vec containing
            # the shifts from the reference to the subsequent image; needed
            # below
            if fn1==ref_fn:
                ref_idx = idx
                
            cdx,cdy = rigid_register(np.load(fn1),np.load(fn2),max_shift=None)
            cdx_vec.append(cdx)
            cdy_vec.append(cdy)

        cdx_vec = np.array(cdx_vec).astype(np.int16)
        cdy_vec = np.array(cdy_vec).astype(np.int16)
        
        np.savetxt(os.path.join(rinfo_directory,'cdx.txt'),cdx_vec)
        np.savetxt(os.path.join(rinfo_directory,'cdy.txt'),cdy_vec)
        np.savetxt(os.path.join(rinfo_directory,'ref_idx.txt'),[ref_idx])
        ref_idx = int(ref_idx)
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(cdx_vec,bins=np.arange(-15.5,16.5))
    plt.subplot(2,1,2)
    plt.hist(cdy_vec,bins=np.arange(-15.5,16.5))

    cdx_vec = cdx_vec - cdx_vec[ref_idx]
    cdy_vec = cdy_vec - cdy_vec[ref_idx]
    
    plt.figure()
    plt.plot(cdx_vec)
    plt.plot(cdy_vec)

    plt.figure()
    plt.plot(np.cumsum(cdx_vec))
    plt.plot(np.cumsum(cdy_vec))
    
    plt.show()


    cum_cdx_vec = np.cumsum(cdx_vec)
    cum_cdy_vec = np.cumsum(cdy_vec)
    
    ref = np.load(ref_fn)
    fref = np.fft.fft2(ref)

    try:
        gabagool
        dx_vec = np.loadtxt(os.path.join(rinfo_directory,'dx.txt')).astype(np.int16)
        dy_vec = np.loadtxt(os.path.join(rinfo_directory,'dy.txt')).astype(np.int16)
    except Exception as e:

        dx_vec = []
        dy_vec = []

        for idx,f in enumerate(fn_list):
            print('%s * %s'%(ref_fn,f))
            tar = np.load(f)
            dx,dy = rigid_register(fref,tar,max_shift,diagnostics,ref_pre_fft=True)
            dx_vec.append(dx)
            dy_vec.append(dy)
            plt.cla()
            plt.plot(dx_vec)
            plt.plot(cum_cdx_vec[idx],'ko')
            plt.plot(dy_vec)
            plt.plot(cum_cdy_vec[idx],'ks')
            plt.pause(.1)
            
        dx_vec = np.array(dx_vec,dtype=np.int16)
        dy_vec = np.array(dy_vec,dtype=np.int16)
        
        np.savetxt(os.path.join(rinfo_directory,'dx.txt'),dx_vec)
        np.savetxt(os.path.join(rinfo_directory,'dy.txt'),dy_vec)

    #dx_vec = -dx_vec
    #dy_vec = -dy_vec
    
    dx_vec = dx_vec - dx_vec.min()
    dy_vec = dy_vec - dy_vec.min()

    sy,sx = ref.shape
    
    osy = sy+np.max(dy_vec)
    osx = sx+np.max(dx_vec)

    for idx,f in enumerate(fn_list):
        out = np.zeros((osy,osx),dtype=np.complex)
        out[dy_vec[idx]:dy_vec[idx]+sy,dx_vec[idx]:dx_vec[idx]+sx] = np.load(f)
        plt.cla()
        plt.imshow(np.abs(out))
        plt.pause(.1)
        


def point_register(ref,tar,max_shift=None,diagnostics=False):
    # bulk shift
    dx,dy = rigid_register(ref,tar,max_shift,diagnoistics)
    sy0,sx0 = ref.shape

    sy,sx = sy+abs(dy),sx+abs(dx)

    ref
    
    
