from matplotlib import pyplot as plt
import numpy as np
import sys,os
from . import bmp_tools
import scipy.optimize as spo
import logging

try:
    from fig2gif import GIF
    can_make_movie = True
except ImportError:
    can_make_movie = False
    
logging.basicConfig(
    level=logging.INFO,
    #format="%(asctime)s [%(levelname)s] %(message)s",
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler("octoblob.log"),
        logging.StreamHandler()
    ]
)

dispersion_3_max = 1.0
dispersion_2_max = 5.0
dispersion_3_min = -dispersion_3_max
dispersion_2_min = -dispersion_2_max
dispersion_3_multiplier = 1e-7
dispersion_2_multiplier = 1e-3

mapping_3_max = 1.0
mapping_2_max = 5.0
mapping_3_min = -mapping_3_max
mapping_2_min = -mapping_2_max
mapping_3_multiplier = 1e-7
mapping_2_multiplier = 1e-3



c3min_default = dispersion_3_min*dispersion_3_multiplier
c3max_default = dispersion_3_max*dispersion_3_multiplier
c2min_default = dispersion_2_min*dispersion_2_multiplier
c2max_default = dispersion_2_max*dispersion_2_multiplier

m3min_default = mapping_3_min*mapping_3_multiplier
m3max_default = mapping_3_max*mapping_3_multiplier
m2min_default = mapping_2_min*mapping_2_multiplier
m2max_default = mapping_2_max*mapping_2_multiplier


c3range = c3max_default-c3min_default
c2range = c2max_default-c2min_default

m3range = m3max_default-m3min_default
m2range = m2max_default-m2min_default

auto_n_points = 6



def max(im):
    return np.max(np.max(im,axis=0))
    #return np.max(im)

def dispersion_ui(raw_data,func,c3min=c3min_default,c3max=c3max_default,c2min=c2min_default,c2max=c2max_default,title=''):

    markersize = 8.0
    
    global points,imaxes,imins
    points = []
    imaxes=[]
    imins=[]
    
    fig,(ax1,ax2) = plt.subplots(1,2)
    plt.suptitle(title)

    
    
    def onclick(event):

        if event.inaxes==ax1:
            return
        
        global points,imaxes,imins
        
        if event.button==1:

            if event.xdata is None and event.ydata is None:
                # clicked outside plot--clear everything
                print('Clearing.')
                points = []
                imaxes = []
                imins = []
                click_points = []
                ax2.cla()
                ax2.set_xlim([c2min,c2max])
                ax2.set_ylim([c3min,c3max])
                plt.draw()
            else:
                xnewclick = event.xdata
                ynewclick = event.ydata
        
                click_points = [(xnewclick,ynewclick)]
                
        elif event.button==3:
            click_points = []
            if len(points)>=2:
                mat = np.array(points)
                c2vals = mat[:,0]
                c3vals = mat[:,1]
                c3start = c3vals.min()
                c3end = c3vals.max()
                c2start = c2vals.min()
                c2end = c2vals.max()
            else:
                c3start = c3min+c3range/float(auto_n_points)/2.0
                c3end = c3max-c3range/float(auto_n_points)/2.0
                c2start = c2min+c2range/float(auto_n_points)/2.0
                c2end = c2max-c2range/float(auto_n_points)/2.0
                
            for x in np.linspace(c2start,c2end,auto_n_points):
                for y in np.linspace(c3start,c3end,auto_n_points):
                    click_points.append((x,y))


        for xnewclick,ynewclick in click_points:
            points.append((xnewclick,ynewclick))

            im = np.abs(func(raw_data,ynewclick,xnewclick))
            #print(np.max(im),np.min(im),np.mean(im))
            # get the max value before scaling
            imax = max(im)
            imaxes.append(imax)

            im = bmp_tools.dbscale(im)

            peak_max = np.max(imaxes)
            peak_min = np.min(imaxes)

            ax1.cla()#plt.cla()
            ax1.imshow(im,aspect='auto',cmap='gray',clim=(40,90))
            
            ax2.cla()
            for p,imax in zip(points,imaxes):
                if imax==peak_max:
                    ax2.plot(p[0],p[1],'ro',markersize=markersize)
                else:
                    peak_rel = (imax-peak_min)/(peak_max-peak_min)
                    b = 1.0-(np.clip(peak_rel,0,.5))
                    ax2.plot(p[0],p[1],'go',markersize=markersize,color=(b,b,b),alpha=0.85)

            ax2.set_xlim([c2min,c2max])
            ax2.set_ylim([c3min,c3max])
            plt.pause(.000001)
                #plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event',onclick)

    #plt.subplot(1,2,2,label='foo')
    ax2.set_xlim([c2min,c2max])
    ax2.set_ylim([c3min,c3max])
    plt.show()
    return points,imaxes

def mapping_dispersion_ui(raw_data,func,m3min=m3min_default,m3max=m3max_default,m2min=m2min_default,m2max=m2max_default,c3min=c3min_default,c3max=c3max_default,c2min=c2min_default,c2max=c2max_default,title=''):

    markersize = 8.0
    
    global m3,m2,c3,c2

    m3 = 0.0
    m2 = 0.0
    c3 = 0.0
    c2 = 0.0
    
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    plt.suptitle(title)
    ax1.set_title('B-scan')
    ax2.set_title('mapping coefficients')
    ax3.set_title('dispersion coefficients')
    
    def onclick(event):

        if event.inaxes==ax1:
            return
        
        global m3,m2,c3,c2
        
        if event.button==1:

            if event.xdata is None and event.ydata is None:
                pass
            else:
                if event.inaxes==ax2:
                    m2 = event.xdata
                    m3 = event.ydata
                if event.inaxes==ax3:
                    c2 = event.xdata
                    c3 = event.ydata

        im = np.abs(func(raw_data,m3,m2,c3,c2))
        im = bmp_tools.dbscale(im)

        ax1.cla()#plt.cla()
        ax1.imshow(im,aspect='auto',cmap='gray',clim=(40,90))
            
        ax2.cla()
        ax2.plot(m2,m3,'ro')
        ax3.cla()
        ax3.plot(c2,c3,'ro')
        
        ax2.set_xlim([m2min,m2max])
        ax2.set_ylim([m3min,m3max])
        ax3.set_xlim([c2min,c2max])
        ax3.set_ylim([c3min,c3max])
        
        plt.pause(.000001)

    cid = fig.canvas.mpl_connect('button_press_event',onclick)

    #plt.subplot(1,2,2,label='foo')
    ax2.set_xlim([m2min,m2max])
    ax2.set_ylim([m3min,m3max])
    ax3.set_xlim([c2min,c2max])
    ax3.set_ylim([c3min,c3max])
    plt.show()
    return m3,m2,c3,c2

#def optimize_mapping_dispersion(raw_data,func,m3min,m3max,m2min,m2max,c3min,c3max,c2min,c2max,title=''):
def stats(im):
    x = im.max()
    n = im.min()
    m = im.mean()
    s = im.std()
    c = 20*np.log10((x-n)/(x+n))
    return '%0.1f (mean); %0.1f (max); %0.1f (min); %0.1f (std); %0.3e (contrast dB)'%(m,x,n,s,c)
    
def optimize_mapping_dispersion(raw_data,func,diagnostics=False,maximum_iterations=200,bounds=None,mode='gradient',show_figures=True,make_movie=False):

    make_movie = make_movie and can_make_movie

    base_image = np.abs(func(raw_data,0,0,0,0))
    
    def norm(im):
        return (im-im.mean())/(im.std())
    def xcorr(im1,im2):
        return np.max(np.real(np.fft.ifft2(np.fft.fft2(norm(im1))*np.conj(np.fft.fft2(norm(im2))))))
    ac = xcorr(base_image,base_image)

    base_lateral_mean_variance = np.var(np.mean(base_image,axis=0))
    base_image_quality = np.mean(np.max(np.diff(base_image,axis=0)))

    try:
        output_directory = os.path.join(diagnostics[0],'dispersion_compensation/optimization')
        os.makedirs(output_directory,exist_ok=True)
    except:
        output_directory = 'optimize_mapping_dispersion_diagnostics'
        os.makedirs(output_directory,exist_ok=True)

    optimization_history = []


    if make_movie:
        mov = GIF(os.path.join(output_directory,'optimization.gif'),fps=30,dpi=50)
    
    def f(coefs):
        m3,m2,c3,c2 = coefs
        im = np.abs(func(raw_data,m3,m2,c3,c2))
        xc = xcorr(im,base_image)/ac

        gradient = np.diff(im,axis=0)
        gradient = np.sort(gradient,axis=0)
        gradient = gradient[-1:,:]

        brights = np.sort(im,axis=0)
        brights = brights[-1:,:]

        if mode=='gradient':
            image_quality = np.nanmedian(np.nanmedian(gradient,axis=0))
        elif mode=='brightness':
            image_quality = np.nanmedian(np.nanmedian(brights,axis=0))
        elif mode=='hybrid':
            image_quality = np.nanmedian(np.nanmedian(gradient,axis=0))+np.nanmedian(np.nanmedian(brights,axis=0))
            
        
        lateral_mean_variance = np.var(np.mean(im,axis=0))

        if xc<0.8:
            image_quality = 1e-10
        if lateral_mean_variance>base_lateral_mean_variance*10.0:
            image_quality = 1e-10
        if image_quality>base_image_quality*10.0:
            image_quality = 1e-10

        if diagnostics:
            plt.subplot(1,2,1)
            plt.cla()
            #plt.imshow(im,cmap='gray')
            plt.imshow(20*np.log10(im),cmap='gray',clim=(40,90))
            plt.text(0,0,'%0.1f (%0.1f)\n%0.1f (%0.1f)\n%0.2f'%(image_quality,base_image_quality,
                                                                  lateral_mean_variance,base_lateral_mean_variance
                                                                  ,xc),ha='left',va='top',fontsize=9,color='g')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1,2,2)
            plt.cla()
            plt.plot([o[0] for o in optimization_history])
            
            if make_movie:
                mov.add(plt.gcf())
                
            plt.pause(.001)

        if image_quality>1 or True:
            optimization_history.append((image_quality,coefs))
        else:
            optimization_history.append((np.nan,coefs))

        logging.info('Optimizer: coefs: %s'%coefs)
        logging.info('Optimizer: B-scan quality: %0.3f'%image_quality)
        logging.info('Optimizer: B-scan stats: %s'%stats(im))

        return 1.0/image_quality
    
    x0 = [0.0,0.0,0.0,0.0]

    res = spo.minimize(f,x0,method='nelder-mead',bounds=bounds,
                       options={'xatol':1e-11,'disp':True,'maxiter':maximum_iterations})

    if make_movie:
        mov.make()
    pre_image = np.abs(func(raw_data,*x0))
    post_image = np.abs(func(raw_data,*res.x))

    #np.save('optimization_pre.npy',pre_image)
    #np.save('optimization_post.npy',post_image)
    
    pre_dB = 20*np.log10(pre_image)
    post_dB = 20*np.log10(post_image)
    
    lin_clim = np.percentile(post_image,(1,99.99))
    dB_clim = (40,90)


    gradient_history = [t[0] for t in optimization_history]

    
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pre_image,cmap='gray',interpolation='none',aspect='auto',clim=lin_clim)
    plt.title('pre optimization, linear')
    plt.subplot(1,2,2)
    plt.imshow(post_image,cmap='gray',interpolation='none',aspect='auto',clim=lin_clim)
    plt.title('post optimization, linear')
    plt.savefig(os.path.join(output_directory,'bscans_linear_%s.png'%mode),dpi=300)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pre_dB,cmap='gray',interpolation='none',aspect='auto',clim=dB_clim)
    plt.title('pre optimization, dB')
    plt.subplot(1,2,2)
    plt.imshow(post_dB,cmap='gray',interpolation='none',aspect='auto',clim=dB_clim)
    plt.title('post optimization, dB')
    plt.savefig(os.path.join(output_directory,'bscans_dB_%s.png'%mode),dpi=300)

    plt.figure()
    plt.plot(gradient_history)
    plt.xlabel('iterations')
    plt.ylabel('bscan max gradient')
    plt.savefig(os.path.join(output_directory,'gradient_history_%s.png'%mode),dpi=100)

    if show_figures:
        plt.show()
    else:
        plt.close('all')
    print(res.x)
    print(stats(post_image))
    return res.x


def optimize_dispersion(raw_data,func,initial_guess,diagnostics=False,maximum_iterations=200,bounds=None,mode='gradient',show_figures=True,make_movie=False):

    # initial_guess doesn't have linear/constant coefs:
    order = len(initial_guess)+1
    
    make_movie = make_movie and can_make_movie

    base_image = np.abs(func(raw_data,initial_guess))
    
    def norm(im):
        return (im-im.mean())/(im.std())
    def xcorr(im1,im2):
        return np.max(np.real(np.fft.ifft2(np.fft.fft2(norm(im1))*np.conj(np.fft.fft2(norm(im2))))))
    ac = xcorr(base_image,base_image)

    base_lateral_mean_variance = np.var(np.mean(base_image,axis=0))
    base_image_quality = np.mean(np.max(np.diff(base_image,axis=0)))

    try:
        output_directory = os.path.join(diagnostics[0],'dispersion_compensation/optimization')
        os.makedirs(output_directory,exist_ok=True)
    except:
        output_directory = 'optimize_mapping_dispersion_diagnostics'
        os.makedirs(output_directory,exist_ok=True)

    optimization_history = []

    if make_movie:
        mov = GIF(os.path.join(output_directory,'optimization.gif'),fps=30,dpi=50)
    
    def f(coefs):
        im = np.abs(func(raw_data,coefs))
        xc = xcorr(im,base_image)/ac

        gradient = np.diff(im,axis=0)
        gradient = np.sort(gradient,axis=0)
        gradient = gradient[-1:,:]

        brights = np.sort(im,axis=0)
        brights = brights[-1:,:]

        if mode=='gradient':
            image_quality = np.nanmedian(np.nanmedian(gradient,axis=0))
        elif mode=='brightness':
            image_quality = np.nanmedian(np.nanmedian(brights,axis=0))
        elif mode=='hybrid':
            image_quality = np.nanmedian(np.nanmedian(gradient,axis=0))+np.nanmedian(np.nanmedian(brights,axis=0))
            
        
        lateral_mean_variance = np.var(np.mean(im,axis=0))

        if xc<0.8:
            image_quality = 1e-10
        if lateral_mean_variance>base_lateral_mean_variance*10.0:
            image_quality = 1e-10
        if image_quality>base_image_quality*10.0:
            image_quality = 1e-10

        if diagnostics:
            plt.subplot(1,2,1)
            plt.cla()
            #plt.imshow(im,cmap='gray')
            plt.imshow(20*np.log10(im),cmap='gray',clim=(40,90))
            plt.text(0,0,'%0.1f (%0.1f)\n%0.1f (%0.1f)\n%0.2f'%(image_quality,base_image_quality,
                                                                  lateral_mean_variance,base_lateral_mean_variance
                                                                  ,xc),ha='left',va='top',fontsize=9,color='g')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(1,2,2)
            plt.cla()
            plt.plot([o[0] for o in optimization_history])
            
            if make_movie:
                mov.add(plt.gcf())
                
            plt.pause(.001)

        if image_quality>1 or True:
            optimization_history.append((image_quality,coefs))
        else:
            optimization_history.append((np.nan,coefs))

        logging.info('Optimizer: coefs: %s'%coefs)
        logging.info('Optimizer: B-scan quality: %0.3f'%image_quality)
        logging.info('Optimizer: B-scan stats: %s'%stats(im))

        return 1.0/image_quality
    
    x0 = initial_guess

    res = spo.minimize(f,x0,method='nelder-mead',bounds=bounds,
                       options={'xatol':1e-11,'disp':True,'maxiter':maximum_iterations})

    if make_movie:
        mov.make()
    pre_image = np.abs(func(raw_data,x0))
    post_image = np.abs(func(raw_data,res.x))

    #np.save('optimization_pre.npy',pre_image)
    #np.save('optimization_post.npy',post_image)
    
    pre_dB = 20*np.log10(pre_image)
    post_dB = 20*np.log10(post_image)
    
    lin_clim = np.percentile(post_image,(1,99.99))
    dB_clim = (40,90)


    gradient_history = [t[0] for t in optimization_history]

    
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pre_image,cmap='gray',interpolation='none',aspect='auto',clim=lin_clim)
    plt.title('pre optimization, linear')
    plt.subplot(1,2,2)
    plt.imshow(post_image,cmap='gray',interpolation='none',aspect='auto',clim=lin_clim)
    plt.title('post optimization, linear')
    plt.savefig(os.path.join(output_directory,'bscans_linear_%s.png'%mode),dpi=300)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pre_dB,cmap='gray',interpolation='none',aspect='auto',clim=dB_clim)
    plt.title('pre optimization, dB')
    plt.subplot(1,2,2)
    plt.imshow(post_dB,cmap='gray',interpolation='none',aspect='auto',clim=dB_clim)
    plt.title('post optimization, dB')
    plt.savefig(os.path.join(output_directory,'bscans_dB_%s.png'%mode),dpi=300)

    plt.figure()
    plt.plot(gradient_history)
    plt.xlabel('iterations')
    plt.ylabel('bscan max gradient')
    plt.savefig(os.path.join(output_directory,'gradient_history_%s.png'%mode),dpi=100)

    if show_figures:
        plt.show()
    else:
        plt.close('all')
    print(res.x)
    print(stats(post_image))
    return res.x
