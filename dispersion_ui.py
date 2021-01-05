from matplotlib import pyplot as plt
import numpy as np
import sys,os
from . import bmp_tools

dispersion_3_max = 10.0
dispersion_2_max = 5.0

dispersion_3_min = -dispersion_3_max

#dispersion_3_multiplier = 1e-16
#dispersion_3_multiplier = 1e-8
dispersion_3_multiplier = 1e-9

dispersion_2_min = -dispersion_2_max

#dispersion_2_multiplier = 1e-10
#dispersion_2_multiplier = 1e-4
dispersion_2_multiplier = 1e-5

c3min = dispersion_3_min*dispersion_3_multiplier
c3max = dispersion_3_max*dispersion_3_multiplier
c2min = dispersion_2_min*dispersion_2_multiplier
c2max = dispersion_2_max*dispersion_2_multiplier

c3range = c3max-c3min
c2range = c2max-c2min

auto_n_points = 6

def max(im):
    return np.median(np.max(im,axis=0))
    #return np.max(im)

def dispersion_ui(raw_data,func,c3min=c3min,c3max=c3max,c2min=c2min,c2max=c2max):

    markersize = 8.0
    
    global points,imaxes,imins
    points = []
    imaxes=[]
    imins=[]
    
    fig,(ax1,ax2) = plt.subplots(1,2)

    
    
    def onclick(event):

        if event.inaxes==ax1:
            return
        
        global points,imaxes,imins

        if event.button==1:
            xnewclick = event.xdata
            ynewclick = event.ydata
        
            click_points = [(xnewclick,ynewclick)]
            
            if xnewclick<c2min or xnewclick>c2max or ynewclick<c3min or ynewclick>c3max:
                print('Clearing.')
                points = []
                imaxes = []
                imins = []
                click_points = []
                ax2.cla()
                ax2.set_xlim([c2min,c2max])
                ax2.set_ylim([c3min,c3max])
                plt.draw()
                
        elif event.button==3:
            click_points = []
            if len(points):

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
            # get the max value before scaling
            imax = max(im)
            imaxes.append(imax)

            im = bmp_tools.logscale(im)

            peak_max = np.max(imaxes)
            peak_min = np.min(imaxes)

            ax1.cla()#plt.cla()
            ax1.imshow(im,aspect='auto',cmap='gray')

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

