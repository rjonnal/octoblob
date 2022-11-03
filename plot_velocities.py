from matplotlib import pyplot as plt
import numpy as np
import sys,os,glob,shutil
import logging
import octoblob.functions as blobf
import octoblob.org_tools as blobo
import pathlib
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9

box_alpha = 0.75
plot_alpha = 0.5
box_linewidth = 0.75
plot_linewidth = 0.75
mplot_alpha = 1.0
mplot_linewidth = 1

def level(im):
    rv = get_level_roll_vec(im)
    return shear(im,rv)

def shear(im,roll_vec):
    out = np.zeros(im.shape)
    for idx,r in enumerate(roll_vec):
        out[:,idx] = np.roll(im[:,idx],r)
    return out

def get_roll_vec(im,row_per_col):
    sy,sx = im.shape
    roll_vec = (np.arange(sx)-sx/2.0)*row_per_col
    roll_vec = np.round(roll_vec).astype(int)
    return roll_vec

def get_level_roll_vec(im,limit=0.1,N=16):
    rpc_vec = np.linspace(-limit,limit,N)
    rotated_profiles = []
    roll_vecs = []
    for rpc in rpc_vec:
        rv = get_roll_vec(im,rpc)
        sheared = shear(im,rv)
        roll_vecs.append(rv)
        rotated_profiles.append(np.mean(sheared,axis=1))

    rotated_profiles = np.array(rotated_profiles)
    rpmax = np.max(rotated_profiles,axis=1)
    widx = np.argmax(rpmax)
    return roll_vecs[widx]

def path2str(f):
    head,tail = os.path.split(f)
    tails = []
    while len(head)>0:
        tails.append(tail)
        head,tail = os.path.split(head)
    tails = tails[::-1]
    return '_'.join(tails)
        
def collect_files(src,dst):
    flist = glob.glob(os.path.join(src,'*'))
    os.makedirs(dst,exist_ok=True)
    
    for f in flist:
        outf = os.path.join(dst,path2str(f))
        shutil.copyfile(f,outf)


def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0

# pay attention to the default value of stim_index, since the b-scans right after stimulus
# determine how the data are displayed to the user; until late 2022, we've been collecting 400
# @ 400 Hz, and the stimulus is delivered 0.25 seconds into the series, i.e. at frame 100; however
# we only process B-scans 80-140, i.e. 50 ms before stimulus through 100 ms after stimulus, and
# thus the stim_index is 20
def plot(folder,stim_index=20):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    phase_slope_flist = glob.glob(os.path.join(folder,'*phase_slope.npy'))
    phase_slope_flist.sort()
    amplitude_flist = glob.glob(os.path.join(folder,'*amplitude.npy'))
    amplitude_flist.sort()

    t = np.arange(len(amplitude_flist))*0.0025-0.04
    
    display_bscan = np.load(amplitude_flist[stim_index])
    dB = 20*np.log10(display_bscan)
    dbclim = np.percentile(dB,(30,99.99))
    
    markersize = 8.0
    
    global rois,click_points,index,abscans,pbscans

    roll_vec = get_level_roll_vec(display_bscan)
    display_bscan = shear(display_bscan,roll_vec)


    abscans = []
    pbscans = []
    for af,pf in zip(amplitude_flist,phase_slope_flist):
        abscans.append(shear(np.load(af),roll_vec))
        pbscans.append(shear(np.load(pf),roll_vec))
    
    abscans = np.array(abscans)
    pbscans = np.array(pbscans)
    
    rois = []
    click_points = []
    index = 0

    fig = plt.figure()
    fig.set_size_inches((3.5,3))
    fig.set_dpi(300)

    ax1 = fig.add_axes([0.03,0.03,.4,0.94])
    ax2 = fig.add_axes([0.6,0.15,0.35,0.82])

    ax1.set_xlim((10,235))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('auto')
    ax1.imshow(20*np.log10(display_bscan),clim=dbclim,cmap='gray',aspect='auto')
    
    ax2.set_ylim((-8,5))
    ax2.set_xlim((-0.05,0.05))
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('$v_{OS}$ ($\mu m$/s)')
    ax2.axvline(0.0,color='g',linestyle='--')
    plt.pause(.0001)


    def draw_rois():
        ax1.clear()
        ax1.set_xlim((10,235))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_aspect('auto')
        ax1.imshow(20*np.log10(display_bscan),clim=dbclim,cmap='gray',aspect='auto')
        for k,roi in enumerate(rois):
            x1,x2 = [a[0] for a in roi[0]]
            z1,z2 = [a[1] for a in roi[0]]
            ax1.plot([x1,x2,x2,x1,x1],[z1,z1,z2,z2,z1],color=colors[k%len(colors)],alpha=box_alpha,linewidth=box_linewidth)

        ax2.clear()
        ax2.set_ylim((-8,5))
        ax2.set_xlim((-0.05,0.05))
        osv_mat = []
        for k,roi in enumerate(rois):
            osv = roi[2]
            osv_mat.append(osv)
            ax2.plot(t,osv,linewidth=plot_linewidth,alpha=plot_alpha,color=colors[k%len(colors)])
        if len(rois)>1:
            osv_mat = np.array(osv_mat)
            mosv = np.mean(osv_mat,axis=0)
            ax2.plot(t,mosv,color='k',alpha=mplot_alpha,linewidth=mplot_linewidth)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('$v_{OS}$ ($\mu m$/s)')
        ax2.axvline(0.0,color='g',linestyle='--')
        plt.pause(.1)
        
    
    def onclick(event):

        global rois,click_points,index,abscans,pbscans

        print(event.button==1)
        print(event.button==2)
        print(event.button==3)
        print()


        if event.button==1:
            if event.xdata is None and event.ydata is None:
                # clicked outside plot--clear everything
                print('Clearing.')
                click_points = []
                rois = []
                draw_rois()
                # ax1.clear()
                # ax1.imshow(20*np.log10(display_bscan),clim=(45,90),cmap='gray',aspect='auto')
                # ax2.clear()
                # ax2.axvline(0.0,color='g',linestyle='--')
                # ax1.set_xticks([])
                # ax1.set_yticks([])
                # plt.pause(.001)

            if event.inaxes==ax1:
                if event.button==1:
                    xnewclick = event.xdata
                    ynewclick = event.ydata
                    click_points.append((int(round(xnewclick)),int(round(ynewclick))))

            if len(click_points)==1:
                #ax1.clear()
                #ax1.imshow(20*np.log10(display_bscan),clim=(45,85),cmap='gray')
                #ax1.plot(click_points[0][0],click_points[0][1],'bo')
                plt.pause(.1)

            if len(click_points)==2:

                x1,x2 = [a[0] for a in click_points]            
                z1,z2 = [a[1] for a in click_points]
                #ax1.clear()
                #ax1.imshow(20*np.log10(display_bscan),clim=(45,90),cmap='gray')
                valid = True
                try:
                    osa,osv,isos_z,cost_z = blobo.extract_layer_velocities_region(abscans,pbscans,x1,x2,z1,z2)
                except Exception as e:
                    print('ROI could not be processed:',e)
                    valid = False
                    click_points = []

                if valid:
                    # osv is now in radians/block
                    # we want it in nm/s
                    # osv * blocks/sec * nm/radian
                    # nm/radian = 1060.0/(2*np.pi)
                    osv = 1e-3*phase_to_nm(osv)/2.5e-3

                    rois.append((click_points,osa,osv,isos_z,cost_z))
                    click_points = []

                    draw_rois()
                    index+=1
                    
        elif event.button==3:
            x = event.xdata
            y = event.ydata
            new_rois = []
            
            for idx,roi in enumerate(rois):
                x1,y1 = roi[0][0]
                x2,y2 = roi[0][1]
                if x1<x<x2 and y1<y<y2:
                    pass
                else:
                    new_rois.append(roi)
            rois = new_rois
            draw_rois()



    def onpress(event):
        global rois,click_points,index
        if event.key=='enter':
            outfolder = os.path.join(folder,'plot_velocities')
            print('Saving results to %s.'%outfolder)
            os.makedirs(outfolder,exist_ok=True)
            np.save(os.path.join(outfolder,'display_bscan.npy'),display_bscan)
            nrois = len(rois)
            fig.savefig(os.path.join(outfolder,'figure_%d_rois.png'%nrois),dpi=300)
            fig.savefig(os.path.join(outfolder,'figure_%d_rois.pdf'%nrois))
            for roi in rois:
                x1,x2 = [a[0] for a in roi[0]]
                z1,z2 = [a[1] for a in roi[0]]
                fnroot = os.path.join(outfolder,'%d_%d_%d_%d_'%(x1,x2,z1,z2))
                np.save(fnroot+'rect_points.npy',roi[0])
                np.save(fnroot+'outer_segment_amplitude.npy',roi[1])
                np.save(fnroot+'outer_segment_velocity.npy',roi[2])
                np.save(fnroot+'isos_z.npy',roi[3])
                np.save(fnroot+'cost_z.npy',roi[4])

            collect_files(outfolder,'./plot_velocities_results')
        elif event.key=='backspace':
            rois = rois[:-1]
            click_points = []
            draw_rois()
                
            
    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    pid = fig.canvas.mpl_connect('key_press_event',onpress)

    #plt.subplot(1,2,2,label='foo')
    plt.show()
    return rois


if __name__=='__main__':


    if len(sys.argv)<2:
        org_folders = pathlib.Path('.').rglob('org')
        org_folders = [str(f) for f in org_folders]
        org_folders.sort()
        for of in org_folders:
            plot(of)
    else:
        folder = sys.argv[1]
        plot(folder)
