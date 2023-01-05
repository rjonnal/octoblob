import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys,shutil
import octoblob.histogram as blobh
import octoblob.diagnostics_tools as blobd
import logging

# some parameters for limiting processing of B-scans
org_stimulus_frame = 100
org_start_frame = 80
org_end_frame = 140


def get_block_filenames(folder,file_filter='*.npy',block_size=5):
    files = sorted(glob.glob(os.path.join(folder,file_filter)))
    first_first = 0
    last_first = len(files)-block_size
    out = []
    for k in range(first_first,last_first):
        out.append(list(files[k] for k in list(range(k,k+block_size))))
    return out

def get_frames(filename_list):
    stack = []
    for f in filename_list:
        stack.append(np.load(f))
    stack = np.array(stack)
    return stack

def compute_phase_velocity(stack,diagnostics=None):
    amplitude_mean = np.mean(np.abs(stack),axis=0)
    phase_stack = np.angle(stack)
    mask = blobh.make_mask(amplitude_mean,diagnostics=diagnostics)
    phase_stack = np.transpose(phase_stack,(1,2,0))
    phase_stack = blobh.bulk_motion_correct(phase_stack,mask,diagnostics=diagnostics)
    phase_stack = np.transpose(phase_stack,(2,0,1))

    stack = np.abs(stack)*np.exp(1j*phase_stack)

    if diagnostics is not None:
        fig = diagnostics.figure()
        plt.subplot(phase_stack.shape[2]+1,1,1)
        plt.imshow(amplitude_mean,aspect='auto',interpolation='none')

        for k in range(phase_stack.shape[2]):
            plt.subplot(phase_stack.shape[2]+1,1,k+2)
            plt.imshow(mask*phase_stack[:,:,k],aspect='auto',interpolation='none')

        diagnostics.save(fig)


def process_org_blocks(folder,block_size=5,signal_threshold_fraction=0.1,histogram_threshold_fraction=0.1,first_start=None,last_start=None,diagnostics=None,redo=False):

    bscan_files = glob.glob(os.path.join(folder,'complex*.npy'))
    bscan_files.sort()

    bscans = []
    for f in bscan_files:
        bscans.append(np.load(f))
    
    N = len(bscan_files)

    if first_start is None:
        first_start = 0
    if last_start is None:
        last_start = N-block_size

    out_folder = os.path.join(folder,'org')
    if os.path.exists(out_folder):
        if not redo:
            sys.exit('%s exists; rerun process_org_blocks with redo=True or delete %s'%(out_folder,out_folder))
        else:
            shutil.rmtree(out_folder)

    os.makedirs(out_folder,exist_ok=True)

    #for start_index in range(first_start,last_start+1):
    def process_block(block,start_index):
        # for each block:
        # 0. an average amplitude bscan
        bscan = np.nanmean(np.abs(block),axis=0)
        outfn = os.path.join(out_folder,'block_%04d_amplitude.npy'%start_index)
        np.save(outfn,bscan)
        
        # 1. create masks for signal statistics and bulk motion correction
        histogram_mask = np.zeros(bscan.shape)
        signal_mask = np.zeros(bscan.shape)

        # there may be nans, so use nanmax
        histogram_threshold = np.nanmax(bscan)*histogram_threshold_fraction
        signal_threshold = np.nanmax(bscan)*signal_threshold_fraction

        histogram_mask = blobh.make_mask(bscan,histogram_threshold,diagnostics)
        signal_mask = blobh.make_mask(bscan,signal_threshold,diagnostics)
        
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

        corrected_block_phase = blobh.bulk_motion_correct(transposed,histogram_mask,diagnostics=diagnostics)

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
        logging.info('process_org_block start %d current %d end %d'%(first_start,start_index,last_start))
        block = bscans[start_index:start_index+block_size]
        block_files = bscan_files[start_index:start_index+block_size]
        logging.info('process_org_block processing files %s'%block_files)
        block = np.array(block)
        process_block(block,start_index)



def get_stacks_folder(folder):
    phase_slope_flist = glob.glob(os.path.join(folder,'*phase_slope.npy'))
    phase_slope_flist.sort()
    amplitude_flist = glob.glob(os.path.join(folder,'*amplitude.npy'))
    amplitude_flist.sort()

    abscans = []
    pbscans = []
    for af,pf in zip(amplitude_flist,phase_slope_flist):
        abscans.append(np.load(af))
        pbscans.append(np.load(pf))

    abscans = np.array(abscans)
    pbscans = np.array(pbscans)
    return abscans,pbscans
        
def extract_layer_velocities_folder(folder,x1,x2,z1,y2):
    abscans,pbscans = get_stacks_folder(folder)
    return extract_layer_velocities(abscans,pbscans,x1,x2,z1,z2)
    


def extract_layer_velocities_region(abscans,pbscans,x1,x2,z1,z2,stim_index=None,refine_z=0):
    if stim_index is None:
        amean = np.mean(abscans,axis=0)
    else:
        amean = np.mean(abscans[stim_index:stim_index+5,:,:],axis=0)

    full_profile = np.mean(amean[:,x1:x2],axis=1)
        
    isos_points = []
    cost_points = []
    #amean[:z1,:] = np.nan
    #amean[z2:,:] = np.nan

    temp = np.nanmean(amean[z1:z2,x1:x2],axis=1)
    mprof = np.zeros(len(temp)+2)
    mprof[1:-1] = temp
    z1-=1
    z2-=1
    
    
    left = mprof[:-2]
    center = mprof[1:-1]
    right = mprof[2:]
    thresh = np.std(amean)

    peaks = np.where(np.logical_and(center>left,center>right))[0]+1

    peakvals = [mprof[p] for p in peaks]
    height_order = np.argsort(peakvals)[::-1]

    peaks = peaks[height_order[:2]]
    peaks.sort()
    #peaks = peaks[:2]

    if False:
        plt.figure()
        plt.plot(mprof)
        for pidx in peaks:
            plt.plot(pidx,mprof[pidx],'ro')
        plt.show()

    dpeak = peaks[1]-peaks[0]

    os_velocity = []
    os_amplitude = []
    isos_z = []
    cost_z = []
    
    for idx in range(abscans.shape[0]):
        isos_p = []
        isos_a = []
        cost_p = []
        cost_a = []
        abscan = abscans[idx]
        pbscan = pbscans[idx]
        isos_z.append([])
        cost_z.append([])
        for x in range(x1,x2):

            if refine_z:
                dzvec = list(range(-refine_z,refine_z+1))
                amps = []
                for dz in dzvec:
                    amps.append(abscan[z1+peaks[0]+dz,x]+abscan[z1+peaks[1]+dz,x])
                dz = dzvec[np.argmax(amps)]
            else:
                dz = 0

                
            zisos = z1+peaks[0]+dz
            zcost = z1+peaks[1]+dz

            isos_p.append(pbscans[idx][zisos,x])
            cost_p.append(pbscans[idx][zcost,x])
            isos_a.append(abscans[idx][zisos,x])
            cost_a.append(abscans[idx][zcost,x])
            isos_z[-1].append(zisos)
            cost_z[-1].append(zcost)
        
        os_p = [c-i for c,i in zip(cost_p,isos_p)]
        os_a = [(c+i)/2.0 for c,i in zip(cost_a,isos_a)]
        os_velocity.append(np.nanmean(os_p))
        os_amplitude.append(np.nanmean(os_a))
        
    os_velocity = -np.array(os_velocity)
    os_amplitude = np.array(os_amplitude)
    isos_z = np.array(isos_z)
    cost_z = np.array(cost_z)
    return os_amplitude,os_velocity,isos_z,cost_z,x1,x2,full_profile


def extract_layer_velocities_lines(abscans,pbscans,x1,x2,z1,z2,x3,x4,z3,z4,stim_index=None,refine_z=0):
    if stim_index is None:
        amean = np.mean(abscans,axis=0)
    else:
        amean = np.mean(abscans[stim_index:stim_index+5,:,:],axis=0)

    full_profile = np.mean(amean[:,x1:x2],axis=1)
        
    isos_points = []
    cost_points = []

    # reconcile x1, x2, x3, and x4 and reduce to x1 and x2 for both
    # lines
    x1 = max(x1,x3)
    x2 = min(x2,x4)

    N = x2-x1
    z_top = np.round(np.linspace(z1,z2,N)).astype(int)
    z_bottom = np.round(np.linspace(z3,z4,N)).astype(int)

    os_velocity = []
    os_amplitude = []
    isos_z = []
    cost_z = []
    
    for idx in range(abscans.shape[0]):
        isos_p = []
        isos_a = []
        cost_p = []
        cost_a = []
        abscan = abscans[idx]
        pbscan = pbscans[idx]
        isos_z.append([])
        cost_z.append([])
        for xidx,x in enumerate(range(x1,x2)):
            zt = z_top[xidx]
            zb = z_bottom[xidx]
            constant_os_length = False
            
            if constant_os_length:
                if refine_z:
                    dzvec = list(range(-refine_z,refine_z+1))
                    amps = []
                    for dz in dzvec:
                        amps.append(abscan[zt+dz,x]+abscan[zb+dz,x])
                    dz = dzvec[np.argmax(amps)]
                else:
                    dz = 0

                zisos = zt+dz
                zcost = zb+dz
            else:
                if refine_z:
                    dzvec = list(range(-refine_z,refine_z+1))
                    amps_isos = []
                    for dz in dzvec:
                        amps_isos.append(abscan[zt+dz,x])
                    dz_isos = dzvec[np.argmax(amps_isos)]
                    
                    amps_cost = []
                    for dz in dzvec:
                        amps_cost.append(abscan[zb+dz,x])
                    dz_cost = dzvec[np.argmax(amps_cost)]
                    
                else:
                    dz_isos = 0
                    dz_cost = 0
                    
                zisos = zt+dz_isos
                zcost = zb+dz_cost
                
            
            isos_p.append(pbscans[idx][zisos,x])
            cost_p.append(pbscans[idx][zcost,x])
            isos_a.append(abscans[idx][zisos,x])
            cost_a.append(abscans[idx][zcost,x])
            isos_z[-1].append(zisos)
            cost_z[-1].append(zcost)
        
        os_p = [c-i for c,i in zip(cost_p,isos_p)]
        os_a = [(c+i)/2.0 for c,i in zip(cost_a,isos_a)]
        os_velocity.append(np.nanmean(os_p))
        os_amplitude.append(np.nanmean(os_a))
        
    os_velocity = -np.array(os_velocity)
    os_amplitude = np.array(os_amplitude)
    isos_z = np.array(isos_z)
    cost_z = np.array(cost_z)
    return os_amplitude,os_velocity,isos_z,cost_z,x1,x2,full_profile




def extract_layer_velocities_rows(abscans,pbscans,x1,x2,z1,z2):
    amean = np.mean(abscans,axis=0)
    isos_points = []
    cost_points = []
    
    os_velocity = []
    os_amplitude = []
    isos_z = []
    cost_z = []
    
    for idx in range(abscans.shape[0]):
        isos_p = []
        isos_a = []
        cost_p = []
        cost_a = []
        abscan = abscans[idx]
        pbscan = pbscans[idx]
        isos_z.append([])
        cost_z.append([])
        for x in range(x1,x2):
            zisos = z1
            zcost = z2
            isos_p.append(pbscans[idx][zisos,x])
            cost_p.append(pbscans[idx][zcost,x])
            isos_a.append(abscans[idx][zisos,x])
            cost_a.append(abscans[idx][zcost,x])
            isos_z[-1].append(zisos)
            cost_z[-1].append(zcost)
            
        os_p = [c-i for c,i in zip(cost_p,isos_p)]
        os_a = [(c+i)/2.0 for c,i in zip(cost_a,isos_a)]
        os_velocity.append(np.nanmean(os_p))
        os_amplitude.append(np.nanmean(os_a))
        
    os_velocity = -np.array(os_velocity)
    os_amplitude = np.array(os_amplitude)
    isos_z = np.array(isos_z)
    cost_z = np.array(cost_z)
    return os_amplitude,os_velocity,isos_z,cost_z

if __name__=='__main__':
    
    unp_files = glob.glob('./examples/*.unp')
    for unp_file in unp_files:
        d = blobd.Diagnostics(unp_file)
        folder = unp_file.replace('.unp','')+'_bscans'
        process_org_blocks(folder)

        
        # block_filenames = get_block_filenames(folder)
        # for bf in block_filenames:
        #     print(bf)
        #     frames = get_frames(bf)
        #     compute_phase_velocity(frames,diagnostics=d)
