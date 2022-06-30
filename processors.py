### put functions here that use octoblob core tools to process files
### generally these should be functions that read files from the disk and write files from the disk,
### while optionally producing diagnostic plots/logs
import os,sys
import logging
from octoblob import config_reader
import octoblob as blob
from matplotlib import pyplot as plt
import numpy as np
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
    handlers=[
        logging.FileHandler("octoblob.log"),
        logging.StreamHandler()
    ]
)


DEFAULT_PARAMETER_DICTIONARY = {}
DEFAULT_PARAMETER_DICTIONARY['bit_shift_right'] = 4
DEFAULT_PARAMETER_DICTIONARY['dtype'] = 'u'
DEFAULT_PARAMETER_DICTIONARY['fbg_position'] = 85
DEFAULT_PARAMETER_DICTIONARY['fbg_region_height'] = 60
DEFAULT_PARAMETER_DICTIONARY['spectrum_start'] = 159
DEFAULT_PARAMETER_DICTIONARY['spectrum_end'] = 1459
DEFAULT_PARAMETER_DICTIONARY['bscan_z1'] = 1000
DEFAULT_PARAMETER_DICTIONARY['bscan_z2'] = 1300
DEFAULT_PARAMETER_DICTIONARY['bscan_x1'] = 0
DEFAULT_PARAMETER_DICTIONARY['bscan_x2'] = 250
DEFAULT_PARAMETER_DICTIONARY['n_skip'] = 0


def add_parameter(d,parameter_name,default_value=None,info=''):

    if default_value is None:
        try:
            default_value = d[parameter_name]
        except:
            pass

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

def setup_processing(filename):
    # setting diagnostics to True will plot/show a bunch of extra information to help
    # you understand why things don't look right, and then quit after the first loop
    # setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look

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

    src_uncropped = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=None)

    test_uncropped = src_uncropped.get_frame(0).astype(np.float)

    plt.figure()
    plt.imshow(test_uncropped,aspect='auto')
    plt.title('raw spectra')
    plt.colorbar()
    plt.pause(.1)
    parameter_dictionary = DEFAULT_PARAMETER_DICTIONARY
    add_parameter(parameter_dictionary,'eye','RE','RE or LE')
    add_parameter(parameter_dictionary,'ecc_horizontal','2.0','degrees, negative for nasal, positive for temporal')
    add_parameter(parameter_dictionary,'ecc_vertical','0.0','degrees, negative for superior, positive for inferior')
    add_parameter(parameter_dictionary,'fbg_position')
    add_parameter(parameter_dictionary,'fbg_region_height')
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
    
    for k in cfg.keys():
        if not k in ['time_stamp']:
            v = cfg[k]
            parameter_dictionary[k] = v

    outfile = get_param_filename(filename)
    outstring = json.dumps(parameter_dictionary,indent=4, sort_keys=True)
    with open(outfile,'w') as fid:
        fid.write(outstring)
        

# process: read a unp file and write complex bscans:
def process_bscans(filename,diagnostics=False,show_processed_data=False):
    # setting diagnostics to True will plot/show a bunch of extra information to help
    # you understand why things don't look right, and then quit after the first loop
    # setting show_processed_data to True will spawn a window that shows you how the b-scans and angiograms look


    param_filename = get_param_filename(filename)
    if not os.path.exists(param_filename):
        sys.exit('Parameter file {param_filename} missing. Please run setup_processing(filename) or copy a compatible parameters json file to {path}'.format(path=os.path.split(filename)[0],param_filename=param_filename))

    with open(param_filename,'r') as fid:
        s = fid.read()
        params = json.loads(s)


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
    if diagnostics_base:
        diagnostics_directory = filename.replace('.unp','')+'_diagnostics'
        os.makedirs(diagnostics_directory,exist_ok=True)

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params['fbg_position'],fbg_region_height=params['fbg_region_height'],spectrum_start=params['spectrum_start'],spectrum_end=params['spectrum_end'],bit_shift_right=params['bit_shift_right'],n_skip=params['n_skip'],dtype=params['dtype'])

    sys.exit()
    if show_processed_data:
        processing_fig = plt.figure(0)

    for frame_index in range(n_slow):
        if frame_index==end_frame:
            break
        if diagnostics_base:
            diagnostics = (diagnostics_directory,frame_index)


        logging.info('frame %d'%frame_index)
            
        frame = src.get_frame(frame_index,diagnostics=diagnostics)
        frame = blob.dc_subtract(frame,diagnostics=diagnostics)
        frame = blob.k_resample(frame,params.mapping_coefficients,diagnostics=diagnostics)
        frame = blob.dispersion_compensate(frame,params.dispersion_coefficients,diagnostics=diagnostics)
        frame = blob.gaussian_window(frame,0.9,diagnostics=diagnostics)
        bscan = blob.spectra_to_bscan(frame,oversampled_size=params.fft_oversampling_size,z1=params.bscan_z1,z2=params.bscan_z2,diagnostics=diagnostics)
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



if __name__=='__main__':
    process_bscans('/home/rjonnal/Dropbox/Data/conventional_org/flash/test_set/16_53_25.unp')
