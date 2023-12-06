import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob
import functions as blobf
from matplotlib.widgets import Button, Slider
import scipy.signal as sps

dB_clims = (40,90)

try:
    bscan_folder = sys.argv[1]
except:
    print('Please supply the bscan folder at the command line, i.e., python mwe_step_03_make_org_blocks.py XX_YY_ZZ_bscans')
    sys.exit()

BLOCK_SIZE = 5 # number of B-scans to use in phase velocity estimation
BSCAN_INTERVAL = 2.5e-3 # time between B-scans

# IMPORTANT: DEFINE T_STIMULUS RELATIVE TO THE B-SCANS BEING PROCESSED!
# Remember that not all the B-scans may have been processed, and sometimes
# we only process scans e.g., 80-140; if this is the case, the stimulus starts
# at 50 ms (i.e., the 100th scan, the 20th in the series, thus 20x2.5ms)
T_STIMULUS = 50e-3

# set these to None for automatic estimation of thresholds
PEAK_THRESHOLD = None #6000 # threshold for detecting IS/OS and COST peaks
ORG_THRESHOLD = None #2000 # threshold that must be met by IS/OS and COST to calculate ORG

Z_SEARCH_HALF_WIDTH = 1 # distance from nominal IS/OS or COST location to search for true peak
AXIAL_AVERAGING_HALF_WIDTH = 1 # 

# range of B-scan that was stimulated
# sampling is 3 um per pixel in fast dimension
# stimulus is 360 um in diameter (120 pixels)
# B-scan is 750 um in length (250 pixels)
# if stimulus is centered on the B-scan, then start and end should be 65 and 185, respectively
STIMULUS_FAST_START = 65
STIMULUS_FAST_END = 185

# parameters shifting histogram method
N_BASE_BINS = 8
N_BIN_SHIFTS = 8
HISTOGRAM_THRESHOLD_FRACTION = 0.05


#################################### End of hard coded parameters #############################

def load_dict(fn):
    with open(fn,'r') as fid:
        s = fid.read()
        d = json.loads(s)
    return d

def save_dict(fn,d):
    s = json.dumps(d)
    with open(fn,'w') as fid:
        fid.write(s)
        
def xcorr(tup):
    ref = tup[0]
    tar = tup[1]
    nxc = np.abs(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))
    pidx = np.argmax(nxc)
    p = np.max(nxc)
    if pidx>len(tar)//2:
        pidx = pidx-len(tar)
    return p,pidx

def dB(arr):
    return 20*np.log10(arr)

def get_yflatten_info(source_volume,serial_flattening=True):
    print('Getting flattening info.')
    
    n_candidates = 10
    ref_candidate_indices = range(0,source_volume.shape[0],source_volume.shape[0]//n_candidates)

    ref_candidate_bscans = []
    for rci in ref_candidate_indices:
        ref_candidate_bscans.append(np.abs(source_volume[rci,:,:]))

    ref_candidates = [np.mean(b,axis=1) for b in ref_candidate_bscans]

    xcmat = np.ones((n_candidates,n_candidates))*np.nan

    for ridx1 in range(n_candidates):
        for ridx2 in range(ridx1,n_candidates):
            p, pidx = xcorr((ref_candidates[ridx1],ref_candidates[ridx2]))
            xcmat[ridx1,ridx2] = p
            xcmat[ridx2,ridx1] = p

    winner = np.argmax(np.sum(xcmat,axis=0))
    ref = ref_candidates[winner]
    ref_index = ref_candidate_indices[winner]

    tars = [np.mean(np.abs(b),axis=1) for b in source_volume]
    refs = [ref]*len(tars)
    tups = list(zip(tars,refs))
    try:
        assert serial_flattening==False
        n_cpus = os.cpu_count()
        p = mp.Pool(n_cpus)
        xcorr_output = p.map(xcorr,tups)
    except Exception as e:
        print(e)
        xcorr_output = []
        for tup in tups:
            xcorr_output.append(xcorr(tup))

    corrs,shifts = zip(*xcorr_output)
    
    return np.array(corrs),np.array(shifts,dtype=int)


def yflatten_to(source_volume,shifts):
    print('Flattening to...')
    
    for y in range(source_volume.shape[0]):
        source_volume[y,:,:] = np.roll(source_volume[y,:,:],shifts[y],axis=0)
        
    return source_volume


def yflatten(source_volume,thresh=0.5,medfilt_kernel=5):
    corrs,shifts = get_yflatten_info(source_volume)
    shifts = sps.medfilt(shifts,medfilt_kernel)
    shifts[np.where(corrs<thresh*np.max(corrs))] = 0
    source_volume = yflatten_to(source_volume,shifts)
    return source_volume



stimulus_index = int(round(T_STIMULUS/BSCAN_INTERVAL))

tag = bscan_folder.replace('_bscans/','')

diagnostics = blobf.Diagnostics(tag)
# diagnostics = False # if you want to skip the diagnostics
stimulated_range = range(STIMULUS_FAST_START,STIMULUS_FAST_END) 


org_folder = os.path.join(bscan_folder,'org')
os.makedirs(org_folder,exist_ok=True)

redo = False

bscan_files = glob.glob(os.path.join(bscan_folder,'complex*.npy'))
bscan_files.sort()

# check that the stimulus index B-scan is the 100th
stimulus_filename = os.path.split(bscan_files[stimulus_index])[-1]
try:
    assert stimulus_filename.find('100.npy')>-1
except AssertionError as ae:
    print(ae)
    print('Expecting the filename ending with ***100.npy to be the stimulus onset B-scan but T_STIMULUS set to %0.2e. which corresponds to file %s.'%(T_STIMULUS,stimulus_filename))
    sys.exit()


bscans = []
for f in bscan_files:
    bscans.append(np.load(f))


# uncomment the following to add random axial movements for testing algorithms:
# bscans = [np.roll(b,np.random.randint(-15,15),axis=0) for b in bscans]

N = len(bscan_files)
t_vec = np.arange(N-BLOCK_SIZE)*BSCAN_INTERVAL-T_STIMULUS
t_vec = t_vec + (BLOCK_SIZE-1)*BSCAN_INTERVAL


def get_z_crop_coords(bscan,inner_border=20,outer_border=0,noise_level=0.05,diagnostics=False):
    prof = np.mean(np.abs(bscan),axis=1)
    thresh = np.max(prof)*noise_level
    valid = np.where(prof>thresh)[0]
    z2,z1 = valid[-1]+outer_border,valid[0]-inner_border
    z2 = min(bscan.shape[0],z2)
    z1 = max(z1,0)
    if diagnostics:
        fig = diagnostics.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(prof)
        ax1.axhline(thresh)
        ax1.axvline(z1)
        ax1.axvline(z2)
        diagnostics.save()
    return z2,z1

def dB(bscan):
    return 20*np.log10(np.abs(bscan))

# average all B-scans and get automatic cropping coordinates
bscan_mean = np.mean(np.abs(np.array(bscans)),axis=0)
crop_z2,crop_z1 = get_z_crop_coords(bscan_mean,diagnostics=diagnostics)

# crop the B-scans to make them easier to work with
bscans = [b[crop_z1:crop_z2,:] for b in bscans]
bscan_mean_cropped = np.mean(np.abs(np.array(bscans)),axis=0)

bscans_arr = np.array(bscans)
bscans_max = np.max(bscans_arr)
if ORG_THRESHOLD is None:
    ORG_THRESHOLD = bscans_max*0.1
if PEAK_THRESHOLD is None:
    PEAK_THRESHOLD = bscans_max*0.1


if diagnostics:
    label = 'bscan_auto_crop'
    fig = diagnostics.figure(label=label,figsize=(8,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(dB(bscan_mean),cmap='gray')
    ax1.set_title('mean bscan before cropping')
    ax2.imshow(dB(bscan_mean_cropped),cmap='gray')
    ax2.set_title('mean bscan after cropping')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    diagnostics.save()


# shear the reference_bscan to flatten, and save the flattening slope
reference_bscan = bscans[stimulus_index]

def shear(bscan,max_roll):
    out = np.zeros(bscan.shape,dtype=complex)
    roll_vec = np.linspace(0,max_roll,bscan.shape[1])
    roll_vec = np.round(roll_vec).astype(int)
    for k in range(bscan.shape[1]):
        out[:,k] = np.roll(bscan[:,k],roll_vec[k])
    return out

def get_xflattening_function(bscan,min_shift=-30,max_shift=30,diagnostics=False):
    shift_range = range(min_shift,max_shift) # [-20, -19, ..... 19, 20]
    peaks = np.zeros(len(shift_range)) # [0, 0, ..... 0, 0]
    profs = []
    for idx,shift in enumerate(shift_range): # iterate through [-20, -19, ..... 19, 20]
        temp = shear(bscan,shift) # shear by -20, then -19, then -18...
        prof = np.mean(np.abs(temp),axis=1) # compute the lateral median
        profs.append(prof)
        peaks[idx] = np.max(prof) # replace the 0 in peaks with whatever the max value is of prof
    # now, find the location of the highest value in peaks, and use that index to find the optimal shift
    optimal_shift = shift_range[np.argmax(peaks)]
    if diagnostics:
        fig = diagnostics.figure(figsize=(6,8))
        ax = fig.subplots(2,1)
        ax[0].imshow(profs,aspect='auto')
        ax[1].plot(shift_range,peaks)
        ax[1].set_xlabel('max shear')
        ax[1].set_ylabel('max profile peak')
        diagnostics.save()
    return lambda bscan: shear(bscan,optimal_shift)

xflatten = get_xflattening_function(reference_bscan,diagnostics=diagnostics)

if diagnostics:
    label = 'bscan_flattening'
    fig = diagnostics.figure(label=label,figsize=(8,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.imshow(dB(bscan_mean),clim=dB_clims,aspect='auto',cmap='gray')
    flattened_reference_bscan = xflatten(reference_bscan)
    ax2.imshow(dB(flattened_reference_bscan),clim=dB_clims,aspect='auto',cmap='gray')
    diagnostics.save()
    
bscans = [xflatten(b) for b in bscans]
reference_bscan = bscans[stimulus_index]

reference_profile = np.mean(np.abs(reference_bscan),axis=1)

bscans = np.array(bscans)

bscans = yflatten(bscans)


class VisualizationState:

    def __init__(self,volume,peak_threshold,bscan_index,search_half_width):
        self.volume = volume
        self.peak_threshold = peak_threshold
        self.bscan_index = bscan_index
        self.search_half_width = search_half_width
        
        self.bscan = np.abs(self.volume[self.bscan_index,:,:])
        
        self.profile = np.mean(self.bscan,axis=1)
    
        temp = np.array([val for val in self.profile])
        temp[np.where(temp<self.peak_threshold)] = 0

        left = temp[:-2]
        center = temp[1:-1]
        right = temp[2:]

        self.peaks = (center>left).astype(int) * (center>right).astype(int)
        self.peaks = np.where(self.peaks)[0]+1
        
        try:
            self.isos_index = self.peaks[0]
            self.cost_index = self.peaks[1]
        except IndexError as ie:
            self.isos_index = 0
            self.cost_index = 0
        
        self.ytop,self.xtop,self.ybottom,self.xbottom = self.get_valid_points()

    def get_valid_points(self):
        yvalid,xvalid = np.where(self.bscan>self.peak_threshold)
        #correct_height = np.where(np.logical_or(np.abs(self.isos_index-yvalid)<=self.search_half_width,
        #                                        np.abs(self.cost_index-yvalid)<=self.search_half_width))[0]
        correct_height = np.where(np.abs(self.isos_index-yvalid)<=self.search_half_width)[0]
        

        
        yvalid = yvalid[correct_height]
        xvalid = xvalid[correct_height]
        correct_x = np.where(np.logical_and(xvalid>=STIMULUS_FAST_START,
                                            xvalid<=STIMULUS_FAST_END))[0]

        yvalid = yvalid[correct_x]
        xvalid = xvalid[correct_x]
        
        yout_top = []
        xout_top = []
        yout_bottom = []
        xout_bottom = []

        print(xvalid)
        for testx in range(np.min(xvalid),np.max(xvalid)+1):
            xsub = xvalid[np.where(xvalid==testx)[0]]
            ysub = yvalid[np.where(xvalid==testx)[0]]
            if len(xsub)==0:
                continue
            points = [self.bscan[y,x] for y,x in zip(ysub,xsub)]

            topwinnery = ysub[np.argmax(points)]
            topwinnerx = xsub[np.argmax(points)]
            ascan = np.array([k for k in self.bscan[:,testx]])
            ascan[:self.cost_index-self.search_half_width] = 0
            ascan[self.cost_index+self.search_half_width+1:] = 0
            bottomwinnerx = topwinnerx
            bottomwinnery = np.argmax(ascan)

            yout_top.append(topwinnery)
            xout_top.append(topwinnerx)
            yout_bottom.append(bottomwinnery)
            xout_bottom.append(bottomwinnerx)

        return yout_top,xout_top,yout_bottom,xout_bottom
        
    def update(self):
        self.bscan = np.abs(self.volume[self.bscan_index,:,:])
        
        self.profile = np.mean(self.bscan,axis=1)
        self.ytop,self.xtop,self.ybottom,self.xbottom = self.get_valid_points()

    def set_bscan_index(self,k):
        self.bscan_index = k
        #self.update()

    def set_isos_index(self,k):
        self.isos_index = k
        #self.update()

    def set_cost_index(self,k):
        self.cost_index = k
        #self.update()

    def set_search_half_width(self,k):
        self.search_half_width = k
        #self.update()

    def set_peak_threshold(self,k):
        self.peak_threshold = k
        #self.update()
        
        
def get_org_parameters(volume,peak_threshold=PEAK_THRESHOLD,bscan_index=stimulus_index,search_half_width=Z_SEARCH_HALF_WIDTH,diagnostics=False):

    vs = VisualizationState(volume,peak_threshold,bscan_index,search_half_width)
    
    # Create the figure and the line that we will manipulate
    fig = plt.figure(figsize=(12,6))
    plot_ax = fig.add_axes([0.1,0.4,0.35,0.55])
    plot_handle = plot_ax.plot(vs.profile)[0]
    img_ax = fig.add_axes([0.55,0.4,0.35,0.55])
    img_handle = img_ax.imshow(dB(vs.bscan),clim=dB_clims,cmap='gray')

    ytop = vs.ytop
    xtop = vs.xtop
    ybottom = vs.ybottom
    xbottom = vs.xbottom

    top_handle = img_ax.plot(xtop,ytop,'bo',markersize=2,alpha=0.5)[0]
    bottom_handle = img_ax.plot(xbottom,ybottom,'go',markersize=2,alpha=0.5)[0]
    
    isos_line_handle = plot_ax.axvline(vs.isos_index)
    cost_line_handle = plot_ax.axvline(vs.cost_index)
    threshold_handle = plot_ax.axhline(vs.peak_threshold,color='r')
    
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.5)

    # Make a horizontal slider to control the cost
    axisos = fig.add_axes([0.25, 0.3, 0.5, 0.02])
    isos_slider = Slider(
        ax=axisos,
        label='ISOS index',
        valmin=0,
        valmax=vs.bscan.shape[0],
        valinit=vs.isos_index,
        valstep = 1
    )

    # Make a second horizontally oriented slider to control the isos
    axcost = fig.add_axes([0.25, 0.26, 0.5, 0.02])
    cost_slider = Slider(
        ax=axcost,
        label='COST index',
        valmin=0,
        valmax=vs.bscan.shape[0],
        valinit=vs.cost_index,
        valstep=1
    )

    axthresh = fig.add_axes([0.25,0.22,0.5,0.02])
    thresh_slider = Slider(
        ax = axthresh,
        label='threshold',
        valmin = np.min(vs.bscan),
        valmax = np.max(vs.bscan),
        valinit = vs.peak_threshold)

    axbscanindex = fig.add_axes([0.25,0.18,0.5,0.02])
    bscan_index_slider = Slider(
        ax = axbscanindex,
        label = 'bscan index',
        valmin = 0,
        valmax = vs.volume.shape[0],
        valinit = vs.bscan_index,
        valstep = 1)

    axsearchwidth = fig.add_axes([0.25,0.14,0.5,0.02])
    search_half_width_slider = Slider(
        ax = axsearchwidth,
        label = 'search half width',
        valmin = 0,
        valmax = 5,
        valinit = vs.search_half_width,
        valstep = 1)
    
    # The function to be called anytime a slider's value changes
    def update(val):
        isos_index = isos_slider.val
        cost_index = cost_slider.val
        peak_threshold = thresh_slider.val
        bscan_index = bscan_index_slider.val
        search_half_width = search_half_width_slider.val

        vs.set_isos_index(isos_index)
        vs.set_cost_index(cost_index)
        vs.set_bscan_index(bscan_index)
        vs.set_peak_threshold(peak_threshold)
        vs.set_search_half_width(search_half_width)
        vs.update()
        
        
        isos_line_handle.set_xdata([vs.isos_index,vs.isos_index])
        cost_line_handle.set_xdata([vs.cost_index,vs.cost_index])
        threshold_handle.set_ydata([vs.peak_threshold,vs.peak_threshold])
        plot_handle.set_ydata(vs.profile)
        
        img_handle.set_data(dB(vs.bscan))
        top_handle.set_xdata(vs.xtop)
        top_handle.set_ydata(vs.ytop)
        bottom_handle.set_xdata(vs.xbottom)
        bottom_handle.set_ydata(vs.ybottom)
        fig.canvas.draw_idle()


    # register the update function with each slider
    isos_slider.on_changed(update)
    cost_slider.on_changed(update)
    thresh_slider.on_changed(update)
    bscan_index_slider.on_changed(update)
    search_half_width_slider.on_changed(update)
    
    # Create a `matplotlib.widgets.Button` to quit the sliders to initial values.
    quitax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    quitbutton = Button(quitax, 'Quit', hovercolor='0.975')

    # Create a `matplotlib.widgets.Button` to save slider values.
    continueax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
    continuebutton = Button(continueax, 'Continue', hovercolor='0.975')


    def quit(event):
        sys.exit()

    quitbutton.on_clicked(quit)

    def cont(event):
        plt.close('all')
    
    continuebutton.on_clicked(cont)
    plt.show()

    return vs


vs = get_org_parameters(bscans,diagnostics=diagnostics)

# In this version of the ORG processing we approach the problem more generally, as follows:

# 1. Identify features in the axial structure; these may be peaks above a certain threshold, or rapid
# changes in slope (e.g., when the COST forms a 'shoulder' in the ROST or RPE peak, rather than a proper
# peak.
# 2. Instead of labeling the features at this point, we treat them generically.
# 2. Record the absolute velocities of these features for every fast location in the B-scan block.
# 3. Record the phase variance of the block as well.


first_start = 0
last_start = N-BLOCK_SIZE
# working with frames 80 - 140; stimulus at frame 100
org = []
block_variances = []
block_t_vec = np.arange(0,BLOCK_SIZE*BSCAN_INTERVAL,BSCAN_INTERVAL)
diagnostic_histogram_count = 0

for start_idx in range(first_start,last_start):
    end_idx = start_idx+BLOCK_SIZE
    block = bscans[start_idx:end_idx]
    vs.set_bscan_index(start_idx)
    vs.update()
    
    block = np.array(block)
    
    # block shape is BLOCK_SIZE, n_depth, n_fast

    # 1. Average the block in time to get an average amplitude B-scan
    unrolled_amplitude = np.nanmean(np.abs(block),axis=0)

    
    amplitude = np.nanmean(np.abs(block),axis=0)


    # segment the amplitude image, using the isos_ref and cost_ref as guides
    isos_depths = []
    cost_depths = []
    fast_locations = []

    fast_locations = vs.xtop
    isos_depths = vs.ytop
    cost_depths = vs.ybottom
    
    if diagnostics:
        fig = diagnostics.figure(label='ISOS_COST_segmentation')
        ax = fig.add_subplot(111)
        
        ax.imshow(dB(amplitude),cmap='gray',clim=dB_clims,aspect='auto')
        ax.plot(fast_locations,isos_depths,'bo',markersize=2)
        ax.plot(fast_locations,cost_depths,'go',markersize=2)
        diagnostics.save(fig)
        #plt.close(fig)

    
    # 2. The next step is bulk-motion correction
    # 2a. Only bright pixels are used for bulk-motion correction. We'll use all pixels that are at least 10%
    #     of the image max.
    mask = np.zeros(amplitude.shape)
    
    mask[amplitude>np.max(amplitude)*HISTOGRAM_THRESHOLD_FRACTION] = 1
    if diagnostics:
        fig = diagnostics.figure(label='bulk_motion_correction_mask')
        ax = fig.add_subplot(111)
        ax.imshow(mask)
        diagnostics.save(fig)

    # 2b. Now we work our way across the B-scan and do bulk motion correction for each set of 5 sister A-scans;
    #     we use the same mask for each B-scan, obviously. And we measure bulk motion relative to the first
    #     B-scan in the block.
    n_bscans,n_depth,n_fast = block.shape
    
    abs_all_counts = {}
    for k in range(n_bscans):
        abs_all_counts[k] = []
        
    rel_all_counts = {}
    for k in range(n_bscans):
        rel_all_counts[k] = []


    phase_slopes = []

    velocity_bscan = np.zeros(mask.shape)
    sister_variances = []
    for f,isos_depth,cost_depth in zip(fast_locations,isos_depths,cost_depths):
        mask_column = mask[:,f]
        abs_reference_ascan = block[0,:,f]
        abs_reference_pixels = abs_reference_ascan[np.where(mask_column)]
        
        for step in range(1,n_bscans):
            rel_reference_ascan = block[step-1,:,f]
            rel_reference_pixels = rel_reference_ascan[np.where(mask_column)]

            target_ascan = block[step,:,f]
            target_pixels = target_ascan[np.where(mask_column)]

            
            abs_phase_shifts = np.angle(target_pixels)-np.angle(abs_reference_pixels)
            
            rel_phase_shifts = np.angle(target_pixels)-np.angle(rel_reference_pixels)

            # Now we have the phase shifts between the target and reference pixels; these should tell us how much
            # the target A-scan has moved relative to the reference A-scan. Typically, this phase shift is corrected
            # by using the resampling histogram method proposed by Makita 2006 "Optical coherence angiography".

            # Mod 2pi so that negative phase values get wrapped into the [0,2pi] range; this is preferable to using
            
            # np.unwrap because if the first phase is negative, unwrap will make them all negative

            # NOTE TO SELF: WHAT IS THE RANGE OF THE PHASE DIFFERENCES? (It's [-2pi,2pi]).
            # mod 2 pi causes any negative values to be wrapped into the [0,2pi] range in the "correct locations"

            
            
            # abs_phase_shifts = abs_phase_shifts%(np.pi*2)
            # rel_phase_shifts = rel_phase_shifts%(np.pi*2)
            
            # abs_phase_shifts = np.unwrap(abs_phase_shifts)
            # rel_phase_shifts = np.unwrap(rel_phase_shifts)

            # The difference between mod and unwrap:
            # start with [-1.9pi, 1.8pi]
            # mod -> [0.1pi, 1.8pi]
            # unwrap -> [-1.9pi, -2.2pi]
            # The question is: are the statistics (e.g. std, var, skew) of the resulting distributions the same?

            
            # to permit negative values to persist we would have to cover the range [-2pi,2pi]
            full_range = 2*np.pi

            base_bin_width = full_range/N_BASE_BINS
            shift_size = base_bin_width/N_BIN_SHIFTS
            #base_bin_starts = np.arange(0,N_BASE_BINS-1)*base_bin_width

            base_bin_starts = np.arange(2*np.pi-full_range,2*np.pi-base_bin_width,base_bin_width)
            base_bin_end = base_bin_starts[-1]+base_bin_width

            abs_resampled_centers = []
            abs_resampled_counts = []
            rel_resampled_centers = []
            rel_resampled_counts = []

            if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                fig = diagnostics.figure(figsize=(6,N_BIN_SHIFTS//3),label='shifting_histograms')
                ax = fig.subplots(N_BIN_SHIFTS+1,1)
                
            for n_shift in range(N_BIN_SHIFTS+1):
                # to use numpy hist we must specify bin edges including the rightmost edge
                bin_edges = np.zeros(N_BASE_BINS)
                bin_edges[:N_BASE_BINS-1] = base_bin_starts+n_shift*shift_size
                bin_edges[-1] = base_bin_end+n_shift*shift_size

                abs_counts,abs_edges = np.histogram(abs_phase_shifts,bins=bin_edges)
                abs_centers = (abs_edges[1:]+abs_edges[:-1])/2.0
                abs_resampled_centers = abs_resampled_centers+list(abs_centers)
                abs_resampled_counts = abs_resampled_counts+list(abs_counts)

                if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                    ax[n_shift].bar(abs_centers,abs_counts,width=base_bin_width,linewidth=1,edgecolor='k')
                    ax[n_shift].set_xlim((0,2*np.pi))
                
                rel_counts,rel_edges = np.histogram(rel_phase_shifts,bins=bin_edges)
                rel_centers = (rel_edges[1:]+rel_edges[:-1])/2.0
                rel_resampled_centers = rel_resampled_centers+list(rel_centers)
                rel_resampled_counts = rel_resampled_counts+list(rel_counts)

            if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                plt.suptitle('bin_shifted_histograms')
                diagnostics.save(ignore_limit=True)

            abs_order = np.argsort(abs_resampled_centers)
            abs_resampled_counts = np.array(abs_resampled_counts)[abs_order]
            abs_resampled_centers = np.array(abs_resampled_centers)[abs_order]
            abs_all_counts[step-1].append(abs_resampled_counts)
            rel_order = np.argsort(rel_resampled_centers)
            rel_resampled_counts = np.array(rel_resampled_counts)[rel_order]
            rel_resampled_centers = np.array(rel_resampled_centers)[rel_order]
            rel_all_counts[step-1].append(rel_resampled_counts)

            if diagnostics and start_idx==first_start and diagnostic_histogram_count<5:
                fig = diagnostics.figure(label='resampled_histogram')
                ax = fig.add_subplot(111)
                ax.bar(abs_resampled_centers,abs_resampled_counts,width=base_bin_width/N_BIN_SHIFTS,linewidth=0.25,edgecolor='k')
                ax.set_xlim((0,2*np.pi))
                ax.set_title('resampled_histogram')
                diagnostics.save(ignore_limit=True)
                diagnostic_histogram_count += 1


            winners = abs_resampled_centers[np.where(abs_resampled_counts==np.max(abs_resampled_counts))]
            winners = np.unwrap(winners)
            ascan_phase_shift = np.median(winners)
            #ascan_phase_shift = abs_resampled_centers[np.argmax(abs_resampled_counts)]

            block[step,:,f] = block[step,:,f] * np.exp(-1j*ascan_phase_shift)
            if False:
                test_pixels = block[step,:,f][np.where(mask_column)]
                print('pre-correction correlation:',np.corrcoef(np.angle(abs_reference_pixels),np.angle(target_pixels))[1,0])
                print('post-correction correlation:',np.corrcoef(np.angle(abs_reference_pixels),np.angle(test_pixels))[1,0])


        sisters = block[:,:,f]
        sister_variance = np.mean(np.var(sisters,axis=0))
        sister_variances.append(sister_variance)
        velocity_ascan = np.zeros(abs_reference_ascan.shape)
        x = np.arange(n_bscans)
        
        for z in range(len(velocity_ascan)):
            if not mask_column[z]:
                continue
            y = np.angle(np.squeeze(sisters[:,z]))
            p = np.polyfit(x,y,1)
            velocity_ascan[z] = p[0]
        velocity_bscan[:,f] = velocity_ascan

        phase_slope_sum = 0.0
        phase_slope_count = 0.0
        for offset in range(-AXIAL_AVERAGING_HALF_WIDTH,AXIAL_AVERAGING_HALF_WIDTH+1):
            # in this method, we multiply complex numbers and compute the angle/argument
            # what is the difference between this and computing the difference between
            # the angles? There is no difference.
            #if ascan[cost_depth+offset]>ORG_THRESHOLD and ascan[isos_depth+offset]>ORG_THRESHOLD:
            if True:
                prod = sisters[:,cost_depth+offset]*np.conj(sisters[:,isos_depth+offset])
                dphase = np.unwrap(np.angle(prod))
                polynomial = np.polyfit(block_t_vec,dphase,1)
                slope = polynomial[0]
                phase_slope_sum = phase_slope_sum + slope
                phase_slope_count += 1
                
        phase_slope = phase_slope_sum/phase_slope_count
        phase_slopes.append(phase_slope)
        
    print(start_idx,np.mean(phase_slopes))
    org.append(-np.mean(phase_slopes))
    block_variances.append(np.mean(sister_variances))
    
    if diagnostics and start_idx>stimulus_index-5 and start_idx<stimulus_index+5:
        fig = diagnostics.figure(label='phase_velocity_b_scan')
        fig.clear()
        ax = fig.add_subplot(111)
        im = ax.imshow(velocity_bscan,clim=[-np.pi,np.pi],cmap='gray')
        x1 = np.min(fast_locations)
        x2 = np.max(fast_locations)
        y1 = np.min(isos_depths)-3
        y2 = np.max(isos_depths)+3
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-')
        plt.colorbar(im)
        diagnostics.save(ignore_limit=True)

org = np.array(org)
org = blobf.phase_to_nm(org)/1000.0 # express in microns/sec

result = np.array([t_vec,org,block_variances])
result = result.T
header = 'First column is time in seconds, second column is OS velocity in microns/sec, third column is block variance.'
np.savetxt(os.path.join(org_folder,'org.txt'),result,header=header)


plt.figure()
plt.plot(1000*t_vec,org)
plt.ylabel('$\Delta L_{OS}$ ($\mu m$/s)')
plt.xlabel('time (ms)')
plt.axvline(0,color='g')
plt.savefig(os.path.join(org_folder,'org.png'))

plt.figure()
plt.plot(1000*t_vec,block_variances)
plt.ylabel('block variance')
plt.xlabel('time (ms)')
plt.axvline(0,color='g')
plt.savefig(os.path.join(org_folder,'block_variance.png'))
plt.show()
