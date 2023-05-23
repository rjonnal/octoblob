from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics_tools
from octoblob import parameters
from octoblob import org_tools
import sys,os,glob
import numpy as np

from octoblob import mapping_dispersion_optimizer as mdo
#from octoblob import dispersion_optimizer as mdo

from octoblob import file_manager
import pathlib


# This example shows how to generate dispersion parameters for a specified UNP dataset using optimization.
data_filename = sys.argv[1]
src = blobf.get_source(data_filename)

try:
    sample_index = int(sys.argv[2])
except IndexError:
    sample_index = src.n_samples//2
    
# Create a diagnostics object for inspecting intermediate processing steps
diagnostics = diagnostics_tools.Diagnostics(data_filename)

# New prototype fbg_align function, which uses cross-correlation instead of feature-
# based alignment of spectra.
# Set a limit on the maximum index where the FBG trough could possibly be located.
# This is a critical parameter, as it avoids cross correlation of spectra based on
# structural information; this would prevent the FBG features from dominating the
# cross-correlation and introduce additional phase noise.
# Correlation threshold is the minimum correlation required to consider two spectra
# to be in phase with one another
def fbg_align(spectra,fbg_max_index=150,correlation_threshold=0.9,diagnostics=None):
    # crop the frame to the FBG region
    f = spectra[:fbg_max_index,:].copy()

    if not diagnostics is None:
        fig = diagnostics.figure(figsize=(6,4))
        axes = fig.subplots(2,2)
        axes[0][0].imshow(f,aspect='auto')
        for k in range(f.shape[1]):
            axes[0][1].plot(f[:,k])

    # group the spectra by amount of shift
    # this step avoids having to perform cross-correlation operations on every
    # spectrum; first, we group them by correlation with one another
    # make a list of spectra to group
    to_do = list(range(f.shape[1]))
    # make a list for the groups of similarly shifted spectra
    groups = []
    ref = 0

    # while there are spectra left to group, do the following loop:
    while(True):
        groups.append([ref])
        to_do.remove(ref)
        for tar in to_do:
            c = np.corrcoef(f[:,ref],f[:,tar])[0,1]
            if c>correlation_threshold:
                groups[-1].append(tar)
                to_do.remove(tar)
        if len(to_do)==0:
            break
        ref = to_do[0]

    subframes = []
    for g in groups:
        subf = f[:,g]
        subframes.append(subf)

    # now decide how to shift the groups of spectra by cross-correlating their means
    # we'll use the first group as the reference group:
    group_shifts = [0]
    ref = np.mean(subframes[0],axis=1)
    # now, iterate through the other groups, compute their means, and cross-correlate
    # with the reference. keep track of the cross-correlation peaks in the list group_shifts
    for taridx in range(1,len(subframes)):
        tar = np.mean(subframes[taridx],axis=1)
        xc = np.fft.ifft(np.fft.fft(ref)*np.fft.fft(tar).conj())
        shift = np.argmax(xc)
        if shift>len(xc)//2:
            shift = shift-len(xc)
        group_shifts.append(shift)

    # now, use the groups and the group_shifts to shift all of the spectra according to their
    # group membership:
    for g,s in zip(groups,group_shifts):
        for idx in g:
            spectra[:,idx] = np.roll(spectra[:,idx],s)
            f[:,idx] = np.roll(f[:,idx],s)

    if not diagnostics is None:
        axes[1][0].imshow(f,aspect='auto')
        for k in range(f.shape[1]):
            axes[1][1].plot(f[:,k])
        diagnostics.save(fig)

    return spectra


def spectra_to_bscan(mdcoefs,spectra,diagnostics=None):
    # only the fbg_align function is called locally (from this script);
    # most of the OCT processing is done by blob functions (blobf.XXXX)
    spectra = fbg_align(spectra,diagnostics=diagnostics)
    spectra = blobf.dc_subtract(spectra,diagnostics=diagnostics)
    spectra = blobf.crop_spectra(spectra,diagnostics=diagnostics)

    if len(mdcoefs)==4:
        spectra = blobf.k_resample(spectra,mdcoefs[:2],diagnostics=diagnostics)
        spectra = blobf.dispersion_compensate(spectra,mdcoefs[2:],diagnostics=None)
    elif len(mdcoefs)==2:
        spectra = blobf.dispersion_compensate(spectra,mdcoefs,diagnostics=None)
        
    spectra = blobf.gaussian_window(spectra,sigma=0.9,diagnostics=None)

    # Now generate the bscan by FFT:
    bscan = np.fft.fft(spectra,axis=0)
    # remove the upper half of the B-scan and leave only the bottom half:
    bscan = bscan[bscan.shape[0]//2:,:]

    # could additionally crop the B-scan if desired;
    # for example, could remove the top 10 rows, bottom 50 rows, and 10 columns
    # from the left and right edges:
    # bscan = bscan[10:-50,10:-10]

    # it; we'll also remove 50 rows near the DC (bottom of the image):
    bscan = bscan[:-50,:]
    
    if not diagnostics is None:
        fig = diagnostics.figure()
        axes = fig.subplots(1,1)
        im = axes.imshow(20*np.log10(np.abs(bscan)),aspect='auto')
        plt.colorbar(im)
        diagnostics.save(fig)
    return bscan


sample = src.get_frame(sample_index)


# modify the next line to use the local spectra_to_bscan function by removing 'blobf.':
coefs = mdo.optimize(sample,spectra_to_bscan,show=False,verbose=False,diagnostics=diagnostics)
print(coefs)

plt.figure()
plt.subplot(1,2,1)
unoptimized = spectra_to_bscan([0,0,0,0],sample)
plt.imshow(20*np.log10(np.abs(unoptimized)),cmap='gray',clim=(40,80))
plt.subplot(1,2,2)
optimized = spectra_to_bscan(coefs,sample)
plt.imshow(20*np.log10(np.abs(optimized)),cmap='gray',clim=(40,80))
plt.suptitle(coefs)
plt.show()
