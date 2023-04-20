from matplotlib import pyplot as plt
import numpy as np
import sys,os,glob,shutil
import logging
import octoblob.functions as blobf
import octoblob.org_tools as blobo
import pathlib
# The index of the processed ORG blocks at which the stimulus was delivered.
# A few cases:
# 1. Typical cone ORG applications. We process blocks B-scans 80 through 140.
#    The stimulus flash is given at B-scan 100, which is the 20th processed
#    B-scan. Thus, stimulus_index=20
# 2. Noise/filtering project. We want to see all the pre-stimulus blocks, thus
#    we process B-scans 0 through 140. The stimulus flash is given at 0.25 s
#    (with a B-scan rate of 400 Hz and period of 2.5 ms), thus the stimulus
#    flash is given at the 100th B-scan, and stimulus_index = 100

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
#plt.rcParams.update({'figure.autolayout': True})

stimulus_index = 20
figure_dpi = 48
figsize_inches = (15,10)

box_alpha = 0.75
box_linewidth = 2.0
box_padding = 3.0

line_alpha = 1.0
line_linewidth = 1.0

org_plot_linewidth = 1.0
org_plot_alpha = 1.0

mean_org_plot_alpha = 1.0
mean_org_plot_linewidth = 1

tlim = (-0.04,0.04) # time limits for plotting ORG in s
zlim = (350,650) # depth limits for profile plot in um

vlim = (-5,5) # velocity limits for plotting in um/s

z_um_per_pixel = 3.0

# refine_z specifies the number of pixels (+/-) over which the
# program may search to identify a local peak. The program begins by asking
# the user to trace line segments through two layers of interest. These layers
# may not be smooth. From one A-scan to the next, the brightest pixel or "peak"
# corresponding to the layer may be displaced axially from the intersection
# of the line segment with the A-scan. refine_z specifies the distance (in either
# direction, above or below that intersection) where the program may search for a
# brighter pixel with which to compute the phase. The optimal setting here will
# largely be determined by how isolated the layer of interest is. For a relatively
# isolated layer, such as IS/OS near the fovea, a large value may be best. For
# closely packed layers such as COST and RPE, smaller values may be useful. The
# user receives immediate feedback from the program's selection of bright pixels
# and can observe whether refine_z is too high (i.e., causing the wrong layer
# to be segmented) or too low (i.e., missing the brightest pixels.
refine_z = 1

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
        shutil.move(f,outf)


def phase_to_nm(phase):
    return phase/(4*np.pi*1.38)*1050.0

def nm_to_phase(nm):
    return nm*(4*np.pi*1.38)/1050.0

# pay attention to the default value of stim_index, since the b-scans right after stimulus
# determine how the data are displayed to the user; until late 2022, we've been collecting 400
# @ 400 Hz, and the stimulus is delivered 0.25 seconds into the series, i.e. at frame 100; however
# we only process B-scans 80-140, i.e. 50 ms before stimulus through 100 ms after stimulus, and
# thus the stim_index is 20
def plot(folder,stim_index=stimulus_index):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    phase_slope_flist = glob.glob(os.path.join(folder,'*phase_slope.npy'))
    phase_slope_flist.sort()
    amplitude_flist = glob.glob(os.path.join(folder,'*amplitude.npy'))
    amplitude_flist.sort()


    # now we load the other data that may be useful for filtering:
    correlations_flist = glob.glob(os.path.join(folder,'*correlations.npy'))
    correlations_flist.sort()

    masked_temporal_variance_flist = glob.glob(os.path.join(folder,'*masked_temporal_variance.npy'))
    masked_temporal_variance_flist.sort()

    phase_slope_fitting_error_flist = glob.glob(os.path.join(folder,'*phase_slope_fitting_error.npy'))
    phase_slope_fitting_error_flist.sort()

    temporal_variance_flist = glob.glob(os.path.join(folder,'*temporal_variance.npy'))
    temporal_variance_flist = [f for f in temporal_variance_flist if f.find('masked')==-1]
    temporal_variance_flist.sort()

    #t = np.arange(len(amplitude_flist))*0.0025-0.24
    t = (-stim_index+np.arange(len(amplitude_flist)))*0.0025+10e-3
    
    display_bscan = np.load(amplitude_flist[stim_index])
    dB = 20*np.log10(display_bscan)
    dbclim = np.percentile(dB,(30,99.99))
    
    markersize = 8.0
    
    global rois,click_points,index,abscans,pbscans,tag,correlations,masked_temporal_variance,phase_slope_fitting_error_bscans,temporal_variance
    
    tag = folder.replace('/','_').replace('\\','_')
    roll_vec = get_level_roll_vec(display_bscan)
    display_bscan = shear(display_bscan,roll_vec)


    abscans = []
    pbscans = []
    correlations = []
    masked_temporal_variance = []
    phase_slope_fitting_error_bscans = []
    temporal_variance = []
    
    for pf,af,cf,mtvf,psfef,tvf in zip(phase_slope_flist,amplitude_flist,correlations_flist,masked_temporal_variance_flist,phase_slope_fitting_error_flist,temporal_variance_flist):
        abscans.append(shear(np.load(af),roll_vec))
        pbscans.append(shear(np.load(pf),roll_vec))
        correlations.append(np.load(cf))
        masked_temporal_variance.append(np.load(mtvf))
        phase_slope_fitting_error_bscans.append(shear(np.load(psfef),roll_vec))
        temporal_variance.append(np.load(tvf))
        #plt.figure()
        #plt.imshow(abscans[-1])
        #plt.show()
        
    abscans = np.array(abscans)

    pbscans = np.array(pbscans)
    correlations = np.array(correlations)
    masked_temporal_variance = np.array(masked_temporal_variance)
    phase_slope_fitting_error_bscans = np.array(phase_slope_fitting_error_bscans)
    temporal_variance = np.array(temporal_variance)

    
    rois = []
    click_points = []
    index = 0

    fig = plt.figure()
    fig.set_size_inches(figsize_inches)
    fig.set_dpi(figure_dpi)
    
    ax1 = fig.add_axes([0.03,0.03,.38,0.94])
    ax2 = fig.add_axes([0.51,0.6,0.38,0.37])
    ax3 = fig.add_axes([0.51,0.1,0.38,0.37])
    
    fig.tight_layout()
    
    ax1.set_xlim((10,235))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('auto')
    ax1.imshow(20*np.log10(display_bscan),clim=dbclim,cmap='gray',aspect='auto')
    
    ax2.set_ylim(vlim)
    ax2.set_xlim(tlim)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('$v$ ($\mu m$/s)')
    ax2.axhline(0,color='k',alpha=0.25)
    
    ax3.set_xlabel('depth ($\mu m$)')
    #ax3.set_xlim(zlim)
    ax3.set_yticks([])
    ax3.set_ylabel('amplitude (ADU)')
    
    ax1.set_xlim((10,235))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('auto')
    ax1.imshow(20*np.log10(display_bscan),clim=dbclim,cmap='gray',aspect='auto')
    
    ax2.axvline(0.0,color='g',linestyle='--')
    plt.pause(.0001)


    def draw_rois():
        ax1.clear()
        ax1.set_xlim((10,235))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_aspect('auto')
        ax1.imshow(20*np.log10(display_bscan),clim=dbclim,cmap='gray',aspect='auto')
        
        ax3.clear()
        ax3.set_xlim(zlim)

        l1zmean = 500
        l2zmean = 500
        for k,roi in enumerate(rois):

            full_profile = roi[7]
            full_profile = full_profile-np.min(full_profile)
            full_profile_pv = np.max(full_profile)

            if k==0:
                offset0 = full_profile_pv*0.2

            offset = offset0*k
            
            z_um = np.arange(len(full_profile))*z_um_per_pixel

            
            #all_prof = np.mean(np.mean(abscans,axis=2),axis=0)
            #com = int(round(np.sum(all_prof*z_um)/np.sum(all_prof)))
            #zlim = (com-200,com+200)

            
            x1 = roi[5]
            x2 = roi[6]

            bx1 = x1-box_padding
            bx2 = x2+box_padding
            
            x = np.arange(x1,x2)

            layer_1_z = roi[3][stim_index,:]
            layer_2_z = roi[4][stim_index,:]

            bz1 = np.min(layer_1_z)-box_padding
            bz2 = np.max(layer_2_z)+box_padding
            
            ax1.plot(x,layer_1_z,color=colors[k%len(colors)],alpha=line_alpha,linewidth=line_linewidth)
            ax1.plot(x,layer_2_z,color=colors[k%len(colors)],alpha=line_alpha,linewidth=line_linewidth)

            ax1.plot([bx1,bx2,bx2,bx1,bx1],[bz1,bz1,bz2,bz2,bz1],alpha=box_alpha,linewidth=box_linewidth)

            ax3.plot(z_um,full_profile-offset,color=colors[k%len(colors)],alpha=line_alpha,linewidth=line_linewidth)

            l1zmean = np.mean(layer_1_z)*z_um_per_pixel
            l2zmean = np.mean(layer_2_z)*z_um_per_pixel
            
            ax3.axvline(l1zmean,color=colors[k%len(colors)],alpha=line_alpha,linewidth=line_linewidth,linestyle=':')
            ax3.axvline(l2zmean,color=colors[k%len(colors)],alpha=line_alpha,linewidth=line_linewidth,linestyle=':')            
            
        ax2.clear()
        ax2.set_ylim(vlim)
        ax2.set_xlim(tlim)
        
        ax3.set_xlabel('depth ($\mu m$)')
        lzmean = (l1zmean+l2zmean)/2.0
        new_zlim = (lzmean-150,lzmean+150)
        ax3.set_xlim(new_zlim)
        ax3.set_yticks([])

        
        osv_mat = []
        layer_amplitude_mean_mat = []
        
        for k,roi in enumerate(rois):
            layer_amplitude_mean = roi[1]
            osv = roi[2]
            
            osv_mat.append(osv)
            layer_amplitude_mean_mat.append(layer_amplitude_mean)
            
            ax2.plot(t,osv,linewidth=org_plot_linewidth,alpha=org_plot_alpha,color=colors[k%len(colors)])

            
        if len(rois)>1:
            osv_mat = np.array(osv_mat)
            layer_amplitude_mean_mat = np.array(layer_amplitude_mean_mat)
            mosv = np.nanmean(osv_mat,axis=0)
            mlayer_amplitude_mean = np.nanmean(layer_amplitude_mean_mat,axis=0)
            
            ax2.plot(t,mosv,color='k',alpha=mean_org_plot_alpha,linewidth=mean_org_plot_linewidth)

        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('$v$ ($\mu m$/s)')
        ax2.axvline(0.0,color='g',linestyle='--')
        ax3.set_ylabel('amplitude (ADU)')

        
        plt.pause(.1)
        
    
    def onclick(event):

        global rois,click_points,index,abscans,pbscans,tag,correlations,masked_temporal_variance,phase_slope_fitting_error_bscans,temporal_variance

        if event.button==1:
            if event.xdata is None and event.ydata is None:
                # clicked outside plot--clear everything
                print('Clearing.')
                click_points = []
                rois = []
                draw_rois()

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
                ax1.plot([x1,x2],[z1,z2],'w-')
                plt.pause(.1)

            if len(click_points)==4:

                x1,x2,x3,x4 = [a[0] for a in click_points]            
                z1,z2,z3,z4 = [a[1] for a in click_points]
                valid = True
                print('x1=%0.1f,x2=%0.1f,z1=%0.1f,z2=%0.1f'%(x1,x2,z1,z2))
                print('x3=%0.1f,x4=%0.1f,z3=%0.1f,z4=%0.1f'%(x3,x4,z3,z4))
                try:

                    if True:
                        layer_amplitude_mean,osv,layer_1_z,layer_2_z,x1,x2,full_profile = blobo.extract_layer_velocities_lines(abscans,pbscans,x1,x2,z1,z2,x3,x4,z3,z4,stim_index=stim_index,refine_z=refine_z)
                    else:
                        layer_amplitude_mean,osv,layer_1_z,layer_2_z,x1,x2,full_profile = blobo.extract_layer_velocities_region(abscans,pbscans,x1,x2,z1,z2,stim_index=stim_index,refine_z=refine_z)
                        
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

                    rois.append((click_points,layer_amplitude_mean,osv,layer_1_z,layer_2_z,x1,x2,full_profile))
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
        global rois,click_points,index,tag
        if event.key=='enter':
            outfolder = os.path.join(folder,'layer_velocities_results')
            print('Saving results to %s.'%outfolder)
            os.makedirs(outfolder,exist_ok=True)
            np.save(os.path.join(outfolder,'display_bscan.npy'),display_bscan)
            nrois = len(rois)
            fx1,fx2,fx3,fx4 = [a[0] for a in rois[0][0]]
            fz1,fz2,fz3,fz4 = [a[1] for a in rois[0][0]]
            froi_tag = '%s_%d_%d_%d_%d_'%(tag,fx1,fx2,fz1,fz3)

            
            fig.savefig(os.path.join(outfolder,'figure_%d_rois %s.png'%(nrois,froi_tag)),dpi=300)
            fig.savefig(os.path.join(outfolder,'figure_%d_rois_%s.pdf'%(nrois,froi_tag)))
            fig.savefig(os.path.join(outfolder,'figure_%d_rois_%s.svg'%(nrois,froi_tag)))
            
            for roi in rois:
                
                x1,x2,x3,x4 = [a[0] for a in roi[0]]
                z1,z2,z3,z4 = [a[1] for a in roi[0]]
                roi_tag = '%s_%d_%d_%d_%d_'%(tag,x1,x2,z1,z3)
                fnroot = os.path.join(outfolder,roi_tag)
                np.save(fnroot+'rect_points.npy',roi[0])
                np.save(fnroot+'amplitude.npy',roi[1])
                np.save(fnroot+'velocity.npy',roi[2])
                np.save(fnroot+'layer_1_z.npy',roi[3])
                np.save(fnroot+'layer_2_z.npy',roi[4])

            collect_files(outfolder,'./layer_velocities_results')
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
        folder = '.'
    else:
        folder = sys.argv[1]


    if os.path.split(folder)[1]=='org':
        plot(folder)
    else:
        org_folders = pathlib.Path(folder).rglob('org')
        org_folders = [str(f) for f in org_folders]
        org_folders.sort()
        for of in org_folders:
            print('Working on %s.'%of)
            try:
                plot(of)
            except IndexError as ie:
                continue
