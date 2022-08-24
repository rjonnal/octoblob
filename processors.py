### put functions here that use octoblob core tools to process files
### some of these should be functions that read files from the disk and write files from the disk,
### while optionally producing diagnostic plots/logs; others read files from the disk and write parameters
### to json

import os,sys,glob,shutil
import logging
from octoblob import config_reader,dispersion_ui
from octoblob import segmentation as seg
from octoblob.volume_tools import Volume, VolumeSeries, Boundaries, show3d
from octoblob.default_parameters import default_parameter_dictionary
import octoblob as blob
from matplotlib import pyplot as plt
import numpy as np
import json
import octoblob.plotting_functions as opf
import scipy.signal as sps
from pathlib import Path

opf.setup_plots(mode='paper',style='seaborn-deep')
color_cycle = opf.get_color_cycle()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler("octoblob.log"),
        logging.StreamHandler()
    ]
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def add_parameter(d,parameter_name,default_value=None,info=''):
    if default_value is None:
        try:
            default_value = d[parameter_name]
        except:
            print(d)
            sys.exit('add_parameter: default value for {parameter_name} was {default_value}; if {default_value} is None, an initial value must be present in the dict'.format(parameter_name=parameter_name,default_value=default_value))

    if len(info)>0:
        info = '(%s)'%info
        
    template = 'Please enter a value for {parameter_name} {info} [{default_value}]: '
    msg = template.format(parameter_name=parameter_name,info=info,default_value=default_value)
    reply = input(msg)
    if len(reply)==0:
        val = default_value
    else:
        val = type(default_value)(reply)
    d[parameter_name] = val

def get_param_filename(filename):
    outfile = filename.replace('.unp','')+'_parameters.json'
    return outfile

def load_dict(filename):
    # load a json file into a dictionary
    with open(filename,'r') as fid:
        dstr = fid.read()
        dictionary = json.loads(dstr)
    return dictionary

def save_dict(filename,d):
    dstr = json.dumps(d,indent=4, sort_keys=True, cls=NpEncoder)
    with open(filename,'w') as fid:
        fid.write(dstr)

def setup_processing(filename,copy_existing=False):

    # try to load the local JSON file to use it as default values
    outfile = get_param_filename(filename)

    # if copy_existing is True, look for a AA_BB_CC_parameters.json file in the current
    # directory, and make a copy for this file
    if copy_existing:
        folder = os.path.split(filename)[0]
        filt = os.path.join(folder,'*_parameters.json')
        sources = glob.glob(filt)
        if len(sources)>0:
            src = sources[0]
            dest = outfile
            if os.path.exists(dest):
                print('%s exists, exiting.'%dest)
            else:
                shutil.copy(src,dest)
                print('Copied %s to %s.'%(src,dest))
            return
    
    # setting diagnostics to True will plot/show a bunch of extra information to help
    # you understand why things don't look right, and then quit after the first loop
    # setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look

    # PARAMETERS FOR RAW DATA SOURCE
    cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))


    try:
        parameter_dictionary = load_dict(outfile)
        for k in default_parameter_dictionary.keys():
            if not k in parameter_dictionary.keys():
                parameter_dictionary[k] = default_parameter_dictionary[k]
                
    except Exception as e:
        parameter_dictionary = default_parameter_dictionary

    n_vol = cfg['n_vol']
    n_slow = cfg['n_slow']
    n_repeats = cfg['n_bm_scans']
    n_fast = cfg['n_fast']
    n_depth = cfg['n_depth']

    # some conversions to comply with old conventions:
    n_slow = n_slow//n_repeats
    n_fast = n_fast*n_repeats

    src_uncropped = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None)

    test_uncropped = src_uncropped.get_frame(0).astype(np.float)

    plt.figure()
    plt.imshow(test_uncropped,aspect='auto')
    plt.title('raw spectra')
    plt.colorbar()
    plt.pause(.1)
    add_parameter(parameter_dictionary,'eye',info='RE or LE')
    add_parameter(parameter_dictionary,'ecc_horizontal',info='degrees, negative for nasal, positive for temporal')
    add_parameter(parameter_dictionary,'ecc_vertical',info='degrees, negative for superior, positive for inferior')
    add_parameter(parameter_dictionary,'fbg_position',info='-1 for no FBG alignment')
    add_parameter(parameter_dictionary,'fbg_region_height',info='ignored if fbg_position is -1')
    add_parameter(parameter_dictionary,'spectrum_start')
    add_parameter(parameter_dictionary,'spectrum_end')
    s1 = parameter_dictionary['spectrum_start']
    s2 = parameter_dictionary['spectrum_end']
    test_uncropped = test_uncropped[s1:s2,:]
    test_uncropped = (test_uncropped.T-test_uncropped.mean(axis=1)).T
    bscan_uncropped = 20*np.log10(np.abs(np.fft.fft(test_uncropped,axis=0)))
    plt.figure()
    plt.imshow(test_uncropped,aspect='auto')
    plt.title('cropped and DC-subtracted spectra')
    plt.colorbar()
    plt.figure()
    plt.imshow(bscan_uncropped,aspect='auto',clim=(50,100))
    plt.title('rough B-scan')
    plt.colorbar()
    plt.pause(.1)
    add_parameter(parameter_dictionary,'bscan_z1')
    add_parameter(parameter_dictionary,'bscan_z2')
    add_parameter(parameter_dictionary,'bscan_x1')
    add_parameter(parameter_dictionary,'bscan_x2')
    add_parameter(parameter_dictionary,'fft_oversampling_size')
    add_parameter(parameter_dictionary,'notes')
    
    for k in cfg.keys():
        if not k in ['time_stamp']:
            v = cfg[k]
            parameter_dictionary[k] = v

    outstring = json.dumps(parameter_dictionary,indent=4, sort_keys=True)
    with open(outfile,'w') as fid:
        fid.write(outstring)
        
def optimize_mapping_dispersion(filename,show_figures=False,mode='gradient',diagnostics=False):
    
    params_filename = get_param_filename(filename)
    params = load_dict(params_filename)
    # PARAMETERS FOR RAW DATA SOURCE
    cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

    n_vol = cfg['n_vol']
    n_slow = cfg['n_slow']
    n_repeats = cfg['n_bm_scans']
    n_fast = cfg['n_fast']
    n_depth = cfg['n_depth']

    # some conversions to comply with old conventions:
    n_slow = n_slow//n_repeats
    n_fast = n_fast*n_repeats

    frame_index = n_slow//2 # choose something from the middle of the first volume
    
    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params['fbg_position'],fbg_region_height=params['fbg_region_height'],spectrum_start=params['spectrum_start'],spectrum_end=params['spectrum_end'],bit_shift_right=params['bit_shift_right'],n_skip=params['n_skip'],dtype=params['dtype'])
    
    def process_for_optimization(frame,m3,m2,c3,c2):
        return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),[m3,m2,0.0,0.0]),[c3,c2,0.0,0.0]),0.9),oversampled_size=params['fft_oversampling_size'],z1=params['bscan_z1'],z2=params['bscan_z2'])
    
    bounds = [(params['m3min'],params['m3max']),
              (params['m2min'],params['m2max']),
              (params['c3min'],params['c3max']),
              (params['c2min'],params['c2max'])]

    if diagnostics:
        diagnostics_pair = (filename.replace('.unp','')+'_diagnostics',frame_index)
    else:
        diagnostics_pair = False
        
    m3,m2,c3,c2 = dispersion_ui.optimize_mapping_dispersion(src.get_frame(frame_index),process_for_optimization,diagnostics=diagnostics_pair,bounds=None,maximum_iterations=200,mode=mode,show_figures=show_figures)

    params['m3'] = m3
    params['m2'] = m2
    params['c3'] = c3
    params['c2'] = c2

    save_dict(params_filename,params)

        
# process: read a unp file and write complex bscans:
def process_bscans(filename,diagnostics=False,show_processed_data=False,start_frame=0,end_frame=np.inf):
    # setting diagnostics to True will plot/show a bunch of extra information to help
    # you understand why things don't look right, and then quit after the first loop
    # setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look

    param_filename = get_param_filename(filename)
    if not os.path.exists(param_filename):
        sys.exit('Parameter file {param_filename} missing. Please run setup_processing(filename) or copy a compatible parameters json file to {path}'.format(path=os.path.split(filename)[0],param_filename=param_filename))

    params = load_dict(param_filename)

    # PARAMETERS FOR RAW DATA SOURCE
    #cfg = config_reader.get_configuration(filename.replace('.unp','.xml'))

    n_vol = params['n_vol']
    n_slow = params['n_slow']
    n_repeats = params['n_bm_scans']
    n_fast = params['n_fast']
    n_depth = params['n_depth']

    # some conversions to comply with old conventions:
    n_slow = n_slow//n_repeats
    n_fast = n_fast*n_repeats

    output_directory_bscans = filename.replace('.unp','')+'_bscans'
    os.makedirs(output_directory_bscans,exist_ok=True)

    output_directory_info = filename.replace('.unp','')+'_info'
    os.makedirs(output_directory_info,exist_ok=True)

    if show_processed_data:
        output_directory_png = filename.replace('.unp','')+'_png'
        os.makedirs(output_directory_png,exist_ok=True)

    diagnostics_base = diagnostics
    diagnostics_directory = filename.replace('.unp','')+'_diagnostics'
    os.makedirs(diagnostics_directory,exist_ok=True)

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params['fbg_position'],fbg_region_height=params['fbg_region_height'],spectrum_start=params['spectrum_start'],spectrum_end=params['spectrum_end'],bit_shift_right=params['bit_shift_right'],n_skip=params['n_skip'],dtype=params['dtype'])

    if show_processed_data:
        processing_fig = plt.figure(0)

    for frame_index in range(start_frame,n_slow):
        if frame_index==end_frame:
            break
        if diagnostics_base or frame_index==0:
            diagnostics = (diagnostics_directory,frame_index)
        else:
            diagnostics = False


        logging.info('processing frame %d'%frame_index)

        mapping_coefficients = [params['m3'],params['m2'],0.0,0.0]
        dispersion_coefficients = [params['c3'],params['c2'],0.0,0.0]

        
        
        frame = src.get_frame(frame_index,diagnostics=diagnostics)
        frame = blob.dc_subtract(frame,diagnostics=diagnostics)
        frame = blob.k_resample(frame,mapping_coefficients,diagnostics=diagnostics)
        frame = blob.dispersion_compensate(frame,dispersion_coefficients,diagnostics=diagnostics)
        frame = blob.gaussian_window(frame,0.9,diagnostics=diagnostics)
        bscan = blob.spectra_to_bscan(frame,oversampled_size=params['fft_oversampling_size'],z1=params['bscan_z1'],z2=params['bscan_z2'],diagnostics=diagnostics)
        bscan_out_filename = os.path.join(output_directory_bscans,'complex_bscan_%05d.npy'%frame_index)
        np.save(bscan_out_filename,bscan)
        
        if show_processed_data:
            png_out_filename = os.path.join(output_directory_png,'bscan_%05d.png'%frame_index)
            
            plt.figure(0)
            processing_fig.clear()
            im = plt.imshow(20*np.log10(np.abs(bscan)),aspect=png_aspect_ratio,cmap='gray',clim=png_dB_clim)
            plt.colorbar(im,fraction=0.03)
            plt.title('bscan dB')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(png_out_filename,dpi=150)
            np.save(bscan_out_filename,bscan)
            
        if diagnostics_base:
            # use plt.close('all') instead of plt.show() if you want to save the diagnostic plots
            # without seeing them
            plt.close('all')
            #plt.show()

        if show_processed_data:
            plt.pause(.001)


def flatten_volume(folder):
    flist = glob.glob(os.path.join(folder,'*.npy'))
    flist.sort()
    N = len(flist)
    
    # grab a section from the middle of the volume to use as a reference
    ref_size = 10
    ref_flist = flist[N//2-ref_size//2:N//2+ref_size//2]
    ref = np.abs(np.load(ref_flist[0])).astype(np.float)
    for f in ref_flist[1:]:
        ref = ref + np.abs(np.load(f)).astype(np.float)
    ref = ref/float(ref_size)
    ref = np.mean(ref,axis=1)

    pre = Volume(folder,use_cache=False)
    plt.figure()
    show3d(pre.volume)
    
    coefs = []
    shifts = []

    out_folder = os.path.join(folder,'flattened')
    os.makedirs(out_folder,exist_ok=True)
    
    for f in flist:
        tar_bscan = np.load(f)
        
        tar = np.mean(np.abs(tar_bscan).astype(np.float),axis=1)
        num = np.fft.fft(tar)*np.conj(np.fft.fft(ref))
        denom = np.abs(num)
        nxc = np.real(np.fft.ifft(num/denom))
        shift = np.argmax(nxc)
        if shift>len(nxc)//2:
            shift = shift-len(nxc)
        shifts.append(shift)
        coefs.append(np.max(nxc))
        logging.info('flatten_volume cross-correlating file %s'%f)
        

    #plt.figure()
    #plt.plot(shifts)

    shifts = sps.medfilt(shifts,9)
    shifts = np.round(-shifts).astype(np.int)
    
    #plt.figure()
    #plt.plot(shifts)
    #plt.show()
        
    for f,shift in zip(flist,shifts):
        tar_bscan = np.load(f)
        tar_bscan = np.roll(tar_bscan,shift,axis=0)
        logging.info('flatten_volume rolling file %s by %d'%(f,shift))
        out_fn = os.path.join(out_folder,os.path.split(f)[1])
        np.save(out_fn,tar_bscan)

    post = Volume(out_folder,use_cache=False)
    plt.figure()
    show3d(post.volume)
    plt.show()
    
    
    

def crop_volumes(folder_list,write=False,threshold_dB=-30,inner_padding=-30,outer_padding=60,dispersion_artifact_size=10,inplace=False):
    
    profs = []
    dB_profs = []
    bscans = []
    dB_bscans = []

    uncropped_bscan_fig = plt.figure()
    for idx,folder in enumerate(folder_list):
        logging.info('crop_volumes working on %s'%folder)
        volume = Volume(folder,use_cache=False)

        reference_index = volume.n_slow//2
        
        try:
            subvolume = volume.get_volume()[reference_index-10:reference_index+10,:,:]
        except Exception as e:
            print(e)
            subvolume = volume.get_volume()

        bscan = np.abs(subvolume).mean(axis=0)
        sz,sx = bscan.shape

        x_stop = sx//2

        prof = np.mean(bscan[:,:x_stop],axis=1)
        #prof = np.abs(volume.get_volume()).mean(axis=2).mean(axis=0)

        profs.append(prof)
        bscans.append(bscan)
        dB_prof = prof/np.max(prof)
        dB_prof = 20*np.log10(dB_prof)
        dB_profs.append(dB_prof)

        dB_bscan = 20*np.log10(bscan)
        dB_bscans.append(dB_bscan)
        plt.plot(dB_prof,label='%d'%idx)

    plt.legend()
    plt.title('uncropped, unshifted profiles')
    plt.axhline(threshold_dB,linestyle='--',color='k')
    opf.despine()

    dB_ref = dB_profs[0]

    bright_idx = np.where(dB_ref>threshold_dB)[0]

    rz1 = bright_idx[0]+inner_padding
    rz2 = bright_idx[-1]+outer_padding

    ref = profs[0]

    shifts = []

    for idx,(folder,tar,bscan,dB_bscan) in enumerate(zip(folder_list,profs,bscans,dB_bscans)):
        minlen = min(len(ref),len(tar))-dispersion_artifact_size
        ref = ref[:minlen]
        tar = tar[:minlen]
        np.save('ref.npy',ref)
        np.save('tar.npy',tar)
        
        # original, non-normalized
        #nxc = np.real(np.fft.ifft(np.fft.fft(tar)*np.conj(np.fft.fft(ref))))

        # better, divide by amplitude:
        num = np.fft.fft(tar)*np.conj(np.fft.fft(ref))
        denom = np.abs(num)
        nxc = np.real(np.fft.ifft(num/denom))

        plt.figure()
        plt.plot(nxc)
        plt.title('nxc %d'%idx)
        
        shift = np.argmax(nxc)
        if shift>len(nxc)//2:
            shift = shift-len(nxc)
        shifts.append(shift)

    shifts = np.array(shifts)
    rz1 = max(0,rz1)
    rz2 = min(len(profs[0]),rz2)

    shifts = shifts-np.min(shifts)
    rz2 = rz2-np.max(shifts)

    cropped_bscan_fig = plt.figure()
    for idx,(folder,tar,bscan,dB_bscan,shift) in enumerate(zip(folder_list,profs,bscans,dB_bscans,shifts)):

        tz1 = rz1+shift
        tz2 = rz2+shift
        if write:
            if inplace:
                out_folder = folder
            else:
                out_folder = os.path.join(folder,'cropped')
                os.makedirs(out_folder,exist_ok=True)
                
            flist = sorted(glob.glob(os.path.join(folder,'complex_bscan*.npy')))
            for f in flist:
                basename = os.path.split(f)[1]
                bscan = np.load(f)
                bscan = bscan[tz1:tz2,:]
                out_f = os.path.join(out_folder,basename)
                np.save(out_f,bscan)
                #plt.cla()
                #plt.imshow(20*np.log10(np.abs(bscan)),cmap='gray')
                #plt.pause(0.001)
                logging.info('crop_volumes cropping %s -> %s.'%(f,out_f))
            Path(os.path.join(out_folder,'cropped')).touch()
            

        else:
            plt.figure(uncropped_bscan_fig.number)
            plt.axvline(tz1,color=color_cycle[idx%len(color_cycle)])
            plt.axvline(tz2,color=color_cycle[idx%len(color_cycle)])

            plt.figure(cropped_bscan_fig.number)
            tar = tar/tar.max()
            prof_dB = 20*np.log10(tar[tz1:tz2])

            plt.plot(prof_dB,label='%d'%idx)
            #plt.plot(tar[tz1:tz2],label='%d'%idx)
            plt.figure()
            plt.imshow(dB_bscan,clim=(40,90),cmap='gray',aspect='auto')
            plt.axhspan(tz1,tz2,color='g',alpha=0.15)
            plt.title(idx)

    if not write:
        plt.figure(cropped_bscan_fig.number)
        plt.axhline(threshold_dB,linestyle='--',color='k')
        plt.legend()
        plt.title("preview of cropped volume profiles\nrun with 'write' as a parameter to perform crop.")
        opf.despine()
        plt.show()
    

def show_labeled_layers(folder,bscan_filter='*amplitude.npy',label_filter='*amplitude_labels.json'):
    bscan_files = glob.glob(os.path.join(folder,bscan_filter))
    bscan_files.sort()
    label_files = glob.glob(os.path.join(folder,label_filter))
    label_files.sort()
    try:
        assert len(bscan_files)==len(label_files)
    except:
        print('%d Bscan files'%len(bscan_files))
        print('%d label files'%len(label_files))
        sys.exit()

    plt.figure()
    for bf,lf in zip(bscan_files,label_files):
        bscan = np.load(bf)
        bscan = 20*np.log10(bscan)
        plt.clf()
        plt.imshow(bscan,clim=(40,90),cmap='gray')
        labeldict = load_dict(lf)
        for k in labeldict.keys():
            x = labeldict[k]['x']
            z = labeldict[k]['z']
            plt.plot(x,z,label=k)
        plt.legend()
        plt.pause(.1)
    plt.close()
        
def label_layers(filename_filter,show=False,labels=None):
    files = glob.glob(filename_filter)
    files.sort()

    bscans = []
    bscan_peaks = []
    bscan_profiles = []
    npeaks_poll = []
    
    for f in files:
        bscan = np.abs(np.load(f))
        bscans.append(bscan)
        peaks,profile = seg.get_peaks(bscan,region=2,width=50)
        if f==files[0] and False:
            plt.figure()
            plt.plot(profile)
            plt.plot(peaks,[profile[p] for p in peaks],'rs')
            plt.savefig(f.replace('.npy','')+'_profile.png')
            plt.show()
        bscan_peaks.append(peaks)
        if np.min(np.diff(peaks))<0:
            sys.exit('processors.label_layers: peaks out of order')
        bscan_profiles.append(profile)
        npeaks_poll.append(len(peaks))
    npeaks = int(round(np.median(npeaks_poll)))

    # check to see if caller has supplied labels; by convention these should be ordered by
    # increasing retinal depth, i.e. inner layers to outer layers
    if not labels is None:
        if not len(labels)==npeaks:
            logging.warning('WARNING. label_layers: %d peaks found, but %d labels passed: %s'%(npeaks,len(labels),labels))
            logging.warning('WARNING. Using only %d peaks and the labels %s.'%(len(labels),labels))
            npeaks = len(labels)
    else:
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        labels = [alphabet[k] for k in range(npeaks)]

    
    # now, iterate through the bscans and peaks and identify any that don't match the
    # peak count; for these, cross-correlate with a template and then use the template
    # peak positions +/- the cc offset

    # use the first bscan with npeaks peaks as a reference
    correct_example_index = npeaks_poll.index(npeaks)
    ref = bscans[correct_example_index]
    refpeaks = bscan_peaks[correct_example_index]
    
    corrected_peaks = []
    for bscan,peaks in zip(bscans,bscan_peaks):
        if len(peaks)==npeaks:
            corrected_peaks.append(np.array(peaks))
        else:
            cross_spectrum = np.fft.fft2(bscan)*np.conj(np.fft.fft2(ref))
            amp = np.abs(cross_spectrum)
            xc = np.real(np.fft.ifft2(cross_spectrum/amp))
            yshift = np.unravel_index(np.argmax(xc),xc.shape)[0]
            peaks = [p+yshift for p in refpeaks]
            corrected_peaks.append(np.array(peaks))
            

    paths = []
    
    for f,bscan,cp in zip(files,bscans,corrected_peaks):
        temp = np.zeros(bscan.shape)
        temp[:] = bscan[:]
        temp = seg.smooth2(temp)
        prof = np.mean(temp[:,:5],axis=1)
        ph = np.array([prof[z] for z in cp])
        ordered_indices = np.argsort(ph)[::-1]
        #cp = cp[ordered_indices]

        ordered_indices = np.argsort(cp)
        labels = [labels[oi] for oi in ordered_indices]

        label_filename = os.path.splitext(f)[0]+'_labels.json'
        
        these_paths = []
        label_dictionary = {}
        these_keys = []
        
        for z,key in zip(cp,labels):
            path = seg.find_path(temp,z,layer_half_width=1)
            paths.append(path)
            these_paths.append(path)
            these_keys.append(key)
            label_dictionary[key] = path
            if show:
                plt.cla()
                plt.imshow(bscan,cmap='gray')
                for p,k in zip(these_paths,these_keys):
                    plt.plot(p['x'],p['z'],label=k)
                    plt.pause(.1)
                plt.legend()
                plt.pause(1)

        save_dict(label_filename,label_dictionary)
        
        if show:
            plt.pause(1)
    if show:
        plt.show()
    

def calculate_band_velocity(folder,reference_layer='IS/OS',target_layer='COST',phase_filter='*phase_slope.npy',label_filter='*amplitude_labels.json'):
    phase_files = glob.glob(os.path.join(folder,phase_filter))
    phase_files.sort()
    label_files = glob.glob(os.path.join(folder,label_filter))
    label_files.sort()
    tar_slope_arr = []
    ref_slope_arr = []
    slope1_arr = []
    slope2_arr = []
    idx_arr = []
    
    for idx,(pf,lf) in enumerate(zip(phase_files,label_files)):
        phase = np.load(pf)
        labels = load_dict(lf)
        assert all([k in labels.keys() for k in [reference_layer,target_layer]])
        tar_x = labels[target_layer]['x']
        tar_z = labels[target_layer]['z']
        ref_x = labels[reference_layer]['x']
        ref_z = labels[reference_layer]['z']

        tar_slope = [phase[z,x] for (z,x) in zip(tar_z,tar_x)]
        ref_slope = [phase[z,x] for (z,x) in zip(ref_z,ref_x)]
        slope = [ts-rs for ts,rs in zip(tar_slope,ref_slope)]

        tar_slope = np.nanmedian(tar_slope)
        ref_slope = np.nanmedian(ref_slope)
        slope1 = np.nanmedian(slope)
        slope2 = tar_slope-ref_slope
        
        tar_slope_arr.append(tar_slope)
        idx_arr.append(idx)
        ref_slope_arr.append(ref_slope)
        slope1_arr.append(slope1)
        slope2_arr.append(slope2)

    plt.figure()
    plt.plot(idx_arr,tar_slope_arr,label='COST')
    plt.plot(idx_arr,ref_slope_arr,label='IS/OS')
    plt.plot(idx_arr,slope1_arr,label='OS1')
    plt.plot(idx_arr,slope2_arr,label='OS2')
    plt.legend()


def show_series(folder,file_filter='*.npy',preproc_func=lambda x: x):
    files = glob.glob(os.path.join(folder,file_filter))
    files.sort()
    plt.figure()
    for f in files:
        im = np.load(f)
        im = preproc_func(im)
        plt.cla()
        plt.imshow(im)
        plt.pause(.1)
    plt.close()

    
def process_org_blocks(folder,block_size=5,signal_threshold_fraction=0.1,histogram_threshold_fraction=0.1,diagnostics=False):
    bscan_files = glob.glob(os.path.join(folder,'complex*.npy'))
    bscan_files.sort()

    bscans = []
    for f in bscan_files:
        bscans.append(np.load(f))
    
    N = len(bscan_files)
    
    first_start = 0
    last_start = N-block_size

    out_folder = os.path.join(folder,'org')
    os.makedirs(out_folder,exist_ok=True)

    diagnostics_folder = os.path.join(out_folder,'diagnostics')
    
    for start_index in range(first_start,last_start+1):
        logging.info('process_org_block start %d current %d end %d'%(first_start,start_index,last_start))
        block = bscans[start_index:start_index+block_size]
        block = np.array(block)

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

        histogram_mask[bscan>histogram_threshold] = 1
        signal_mask[bscan>signal_threshold] = 1
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

        if diagnostics:
            diagnostics_flag = (diagnostics_folder,start_index)
        else:
            diagnostics_flag = False

        corrected_block_phase = blob.bulk_motion_correct(transposed,histogram_mask,diagnostics=diagnostics_flag)
        if diagnostics:
            plt.show()
            plt.close('all')

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
                phase = phase%(2*np.pi)
                phase = np.unwrap(phase)
                poly = np.polyfit(t,phase,1)
                slope = poly[1]
                fit = np.polyval(poly,t)
                err = np.sqrt(np.mean((fit-phase)**2))
                slopes[z,x] = slope
                fitting_error[z,x] = err
        outfn = os.path.join(out_folder,'block_%04d_phase_slope.npy'%start_index)
        np.save(outfn,slopes)
        outfn = os.path.join(out_folder,'block_%04d_phase_slope_fitting_error.npy'%start_index)
        np.save(outfn,fitting_error)
                
        
if __name__=='__main__':
    process_bscans('/home/rjonnal/Dropbox/Data/conventional_org/flash/test_set/16_53_25.unp')
