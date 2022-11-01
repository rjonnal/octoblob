from matplotlib import pyplot as plt
import numpy as np
import sys,os,glob
import logging
import octoblob.functions as blobf


# pay attention to the default value of stim_index, since the b-scans right after stimulus
# determine how the data are displayed to the user; until late 2022, we've been collecting 400
# @ 400 Hz, and the stimulus is delivered 0.25 seconds into the series, i.e. at frame 100; however
# we only process B-scans 80-140, i.e. 50 ms before stimulus through 100 ms after stimulus, and
# thus the stim_index is 20
def plot(folder,stim_index=20):


    phase_slope_flist = glob.glob(os.path.join(folder,'*phase_slope.npy'))
    phase_slope_flist.sort()
    amplitude_flist = glob.glob(os.path.join(folder,'*amplitude.npy'))
    amplitude_flist.sort()

    display_bscan = np.load(amplitude_flist[stim_index])
    
    markersize = 8.0
    
    global points,click_points
    points = []
    click_points = []
    
    fig,(ax1,ax2) = plt.subplots(1,2)
    #plt.suptitle(title)

    ax1.imshow(20*np.log10(display_bscan),clim=(45,85),cmap='gray')
    
    
    def onclick(event):

        global points,click_points
        
        if event.xdata is None and event.ydata is None:
            # clicked outside plot--clear everything
            print('Clearing.')
            click_points = []
            
        if event.inaxes==ax1:
            if event.button==1:
                xnewclick = event.xdata
                ynewclick = event.ydata
                click_points.append((int(round(xnewclick)),int(round(ynewclick))))
                print(click_points)
                
        if len(click_points)==1:
            ax1.clear()
            ax1.imshow(20*np.log10(display_bscan),clim=(45,85),cmap='gray')
            ax1.plot(click_points[0][0],click_points[0][1],'bo')
            plt.pause(.1)
                
        elif event.button==3:
            pass

        
        if len(click_points)==2:

            x1,x2 = [a[0] for a in click_points]
            z1,z2 = [a[1] for a in click_points]
            ax1.plot(x2,z2,'bo')
            ax1.plot([x1,x2,x2,x1,x1],[z1,z1,z2,z2,z1],'y-')
            plt.pause(.1)
            click_points = []
            osa,osv = blobf.extract_layer_velocities(folder,x1,x2,z1,z2)
            ax2.plot(osv)
            plt.pause(.1)

    cid = fig.canvas.mpl_connect('button_press_event',onclick)

    #plt.subplot(1,2,2,label='foo')
    plt.show()
    return points


if __name__=='__main__':
    folder = sys.argv[1]
    plot(folder)
