from matplotlib import pyplot as plt
import numpy as np
import sys,os
from . import bmp_tools
import scipy.optimize as spo

dispersion_3_max = 1.0
dispersion_2_max = 5.0

dispersion_3_min = -dispersion_3_max

#dispersion_3_multiplier = 1e-16
#dispersion_3_multiplier = 1e-8
dispersion_3_multiplier = 1e-8

dispersion_2_min = -dispersion_2_max

#dispersion_2_multiplier = 1e-10
#dispersion_2_multiplier = 1e-4
dispersion_2_multiplier = 1e-4

c3min_default = dispersion_3_min*dispersion_3_multiplier
c3max_default = dispersion_3_max*dispersion_3_multiplier
c2min_default = dispersion_2_min*dispersion_2_multiplier
c2max_default = dispersion_2_max*dispersion_2_multiplier

c3range = c3max_default-c3min_default
c2range = c2max_default-c2min_default

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

def mapping_dispersion_ui(raw_data,func,m3min,m3max,m2min,m2max,c3min,c3max,c2min,c2max,title=''):

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
def optimize_mapping_dispersion(raw_data,func,diagnostics=False):

    base_image = np.abs(func(raw_data,0,0,0,0))
    
    def norm(im):
        return (im-im.mean())/(im.std())
    def xcorr(im1,im2):
        return np.max(np.real(np.fft.ifft2(np.fft.fft2(norm(im1))*np.conj(np.fft.fft2(norm(im2))))))
    ac = xcorr(base_image,base_image)

    base_lateral_mean_variance = np.var(np.mean(base_image,axis=0))
    base_max_gradient = np.mean(np.max(np.diff(base_image,axis=0)))


    output_directory = 'optimize_mapping_dispersion_diagnostics'
    os.makedirs(output_directory,exist_ok=True)

    optimization_history = []
    
    def f(coefs):
        m3,m2,c3,c2 = coefs
        im = np.abs(func(raw_data,m3,m2,c3,c2))
        xc = xcorr(im,base_image)/ac
        
        max_gradient = np.mean(np.max(np.diff(im,axis=0),axis=0))
        
        lateral_mean_variance = np.var(np.mean(im,axis=0))

        if xc<0.8:
            max_gradient = 1e-10
        if lateral_mean_variance>base_lateral_mean_variance*10.0:
            max_gradient = 1e-10
        if max_gradient>base_max_gradient*10.0:
            max_gradient = 1e-10

        if diagnostics:
            plt.cla()
            plt.imshow(im,cmap='gray')
            #plt.imshow(20*np.log10(im),cmap='gray',clim=(40,80))
            plt.text(0,0,'%0.1f (%0.1f) / %0.1f (%0.1f) / %0.2f'%(max_gradient,base_max_gradient,
                                      lateral_mean_variance,base_lateral_mean_variance
                                      ,xc),ha='left',va='top',fontsize=12,color='y')
            plt.pause(.001)

        if max_gradient>1:
            optimization_history.append((max_gradient,coefs))
        else:
            optimization_history.append((np.nan,coefs))

        return 1.0/max_gradient

    x0 = [0.0,0.0,0.0,0.0]

    res = spo.minimize(f,x0,method='nelder-mead',
                       options={'xatol':1e-11,'disp':True})

    pre_image = np.abs(func(raw_data,*x0))
    post_image = np.abs(func(raw_data,*res.x))

    pre_dB = 20*np.log10(pre_image)
    post_dB = 20*np.log10(post_image)
    
    lin_clim = np.percentile(post_image,(20,99.9))
    dB_clim = (40,85)


    gradient_history = [t[0] for t in optimization_history]
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pre_image,cmap='gray',interpolation='none',aspect='auto',clim=lin_clim)
    plt.title('pre optimization, linear')
    plt.subplot(1,2,2)
    plt.imshow(post_image,cmap='gray',interpolation='none',aspect='auto',clim=lin_clim)
    plt.title('post optimization, linear')
    plt.savefig(os.path.join(output_directory,'bscans_linear.png'),dpi=300)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pre_dB,cmap='gray',interpolation='none',aspect='auto',clim=dB_clim)
    plt.title('pre optimization, dB')
    plt.subplot(1,2,2)
    plt.imshow(post_dB,cmap='gray',interpolation='none',aspect='auto',clim=dB_clim)
    plt.title('post optimization, dB')
    plt.savefig(os.path.join(output_directory,'bscans_dB.png'),dpi=300)

    plt.figure()
    plt.plot(gradient_history)
    plt.xlabel('iterations')
    plt.ylabel('bscan max gradient')
    plt.savefig(os.path.join(output_directory,'gradient_history.png'),dpi=100)
    
    plt.show()
    print(res.x)
    return res.x
