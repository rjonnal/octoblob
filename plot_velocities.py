from matplotlib import pyplot as plt
import numpy as np
import sys,os,glob
import logging


def plot(folder):


    phase_slope_flist = glob.glob(os.path.join(folder,'*phase_slope.npy'))
    phase_slope_flist.sort()
    amplitude_flist = glob.glob(os.path.join(folder,'*amplitude.npy'))
    amplitude_flist.sort()
    

    print(phase_slope_flist)
    
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


if __name__=='__main__':
    folder = sys.argv[1]
    plot(folder)
