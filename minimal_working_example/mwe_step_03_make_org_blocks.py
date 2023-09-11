import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys,os,glob
import functions as blobf

dB_clims = (40,90)

try:
    bscan_folder = sys.argv[1]
except:
    print('Please supply the bscan folder at the command line, i.e., python mwe_step_03_make_org_blocks.py XX_YY_ZZ_bscans')
    sys.exit()

show_bscans = True
block_size = 5
signal_threshold_fraction = 0.1
histogram_threshold_fraction = 0.1
out_folder = os.path.join(bscan_folder,'org')
redo = False


bscan_files = glob.glob(os.path.join(bscan_folder,'complex*.npy'))
bscan_files.sort()

os.makedirs(out_folder,exist_ok=True)


bscans = []
for f in bscan_files:
    bscans.append(np.load(f))

N = len(bscan_files)

first_start = 0
last_start = N-block_size

def process_block(block,start_index):
    # for each block:
    # 0. an average amplitude bscan
    bscan = np.nanmean(np.abs(block),axis=0)
    outfn = os.path.join(out_folder,'block_%04d_amplitude.npy'%start_index)
    np.save(outfn,bscan)

    if show_bscans:
        plt.cla()
        plt.imshow(20*np.log10(bscan),cmap='gray',clim=dB_clims)
        plt.gca().set_title('Block %d of %d'%(start_index,last_start))
        plt.pause(0.00001)
        
    
    # 1. create masks for signal statistics and bulk motion correction
    histogram_mask = np.zeros(bscan.shape)
    signal_mask = np.zeros(bscan.shape)

    # there may be nans, so use nanmax
    histogram_threshold = np.nanmax(bscan)*histogram_threshold_fraction
    signal_threshold = np.nanmax(bscan)*signal_threshold_fraction

    histogram_mask = blobf.make_mask(bscan,histogram_threshold)
    signal_mask = blobf.make_mask(bscan,signal_threshold)

    outfn = os.path.join(out_folder,'block_%04d_signal_mask.npy'%start_index)
    np.save(outfn,signal_mask)
    outfn = os.path.join(out_folder,'block_%04d_histogram_mask.npy'%start_index)
    np.save(outfn,histogram_mask)


    # 3. do bulk-motion correction on block:
    block_phase = np.angle(block)

    # transpose dimension b/c bulk m.c. requires the first two
    # dims to be depth and x, and the third dimension to be
    # repeats
    transposed = np.transpose(block_phase,(1,2,0))

    corrected_block_phase = blobf.bulk_motion_correct(transposed,histogram_mask)

    corrected_block_phase = np.transpose(corrected_block_phase,(2,0,1))
    block = np.abs(block)*np.exp(1j*corrected_block_phase)

    # 4. estimate(s) of correlation of B-scans (single values)
    corrs = []
    for im1,im2 in zip(block[:-1],block[1:]):
        corrs.append(np.corrcoef(np.abs(im1).ravel(),np.abs(im2).ravel())[0,1])

    outfn = os.path.join(out_folder,'block_%04d_correlations.npy'%start_index)
    np.save(outfn,corrs)

    # 5. temporal variance of pixels--all pixels and bright pixels (single values)
    varim = np.nanvar(np.abs(block),axis=0)
    var = np.nanmean(varim)
    var_masked = np.nanmean(varim[np.where(signal_mask)])
    outfn = os.path.join(out_folder,'block_%04d_temporal_variance.npy'%start_index)
    np.save(outfn,var)
    outfn = os.path.join(out_folder,'block_%04d_masked_temporal_variance.npy'%start_index)
    np.save(outfn,var_masked)


    # 6. phase slopes and residual fitting error for all pixels (2D array)

    slopes = np.ones(bscan.shape)*np.nan
    fitting_error = np.ones(bscan.shape)*np.nan

    st,sz,sx = corrected_block_phase.shape
    t = np.arange(st)

    for z in range(sz):
        for x in range(sx):
            if not signal_mask[z,x]:
                continue
            phase = corrected_block_phase[:,z,x]
            # bug 0: line below does not exist in original ORG processing code:
            #phase = phase%(2*np.pi)

            phase = np.unwrap(phase)
            poly = np.polyfit(t,phase,1)

            # bug 1: line below used to say poly[1]!
            slope = poly[0]
            fit = np.polyval(poly,t)
            err = np.sqrt(np.mean((fit-phase)**2))
            slopes[z,x] = slope
            fitting_error[z,x] = err
    outfn = os.path.join(out_folder,'block_%04d_phase_slope.npy'%start_index)
    np.save(outfn,slopes)
    outfn = os.path.join(out_folder,'block_%04d_phase_slope_fitting_error.npy'%start_index)
    np.save(outfn,fitting_error)

for start_index in range(first_start,last_start+1):
    # look to see if this block has already been calculated; unless redo is True,
    # if it has, then skip
    test_fn = os.path.join(out_folder,'block_%04d_phase_slope_fitting_error.npy'%start_index)
    if os.path.exists(test_fn) and not redo:
        continue

    block = bscans[start_index:start_index+block_size]
    block_files = bscan_files[start_index:start_index+block_size]
    block = np.array(block)
    process_block(block,start_index)
