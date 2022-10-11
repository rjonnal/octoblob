def optimize_mapping_dispersion(filename,show_figures=False,mode='gradient',diagnostics=False,frame_index=0):
    
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
        
    m3,m2,c3,c2 = dispersion_tools.optimize_mapping_dispersion(src.get_frame(frame_index),process_for_optimization,diagnostics=diagnostics_pair,bounds=None,maximum_iterations=200,mode=mode,show_figures=show_figures)

    params['m3'] = m3
    params['m2'] = m2
    params['c3'] = c3
    params['c2'] = c2

    save_dict(params_filename,params)


def optimize_dispersion(filename,show_figures=False,mode='gradient',diagnostics=False,frame_index=0):
    
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

    dispersion_initial = params['dispersion_initial']
    
    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params['fbg_position'],fbg_region_height=params['fbg_region_height'],spectrum_start=params['spectrum_start'],spectrum_end=params['spectrum_end'],bit_shift_right=params['bit_shift_right'],n_skip=params['n_skip'],dtype=params['dtype'])
    
    def process_for_optimization(frame,coefs):
        all_coefs = list(coefs)+[0.0,0.0]
        return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.dc_subtract(frame),all_coefs),0.9),oversampled_size=params['fft_oversampling_size'],z1=params['bscan_z1'],z2=params['bscan_z2'])
    
    bounds = zip(params['cmin'],params['cmax'])

    if diagnostics:
        diagnostics_pair = (filename.replace('.unp','')+'_diagnostics',frame_index)
    else:
        diagnostics_pair = False
        
    coefs = dispersion_tools.optimize_dispersion(src.get_frame(frame_index),process_for_optimization,dispersion_initial,diagnostics=diagnostics_pair,bounds=None,maximum_iterations=200,mode=mode,show_figures=show_figures)

    params['dispersion_optimized'] = coefs
    save_dict(params_filename,params)


def manual_mapping_dispersion(filename,frame_index=0):
    
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

    src = blob.OCTRawData(filename,n_vol,n_slow,n_fast,n_depth,n_repeats,fbg_position=params['fbg_position'],fbg_region_height=params['fbg_region_height'],spectrum_start=params['spectrum_start'],spectrum_end=params['spectrum_end'],bit_shift_right=params['bit_shift_right'],n_skip=params['n_skip'],dtype=params['dtype'])
    
    def process_for_optimization(frame,m3,m2,c3,c2):
        return blob.spectra_to_bscan(blob.gaussian_window(blob.dispersion_compensate(blob.k_resample(blob.dc_subtract(frame),[m3,m2,0.0,0.0]),[c3,c2,0.0,0.0]),0.9),oversampled_size=params['fft_oversampling_size'],z1=params['bscan_z1'],z2=params['bscan_z2'])
    
    m3,m2,c3,c2 = dispersion_tools.mapping_dispersion_tools(src.get_frame(frame_index),process_for_optimization)

    params['m3'] = m3
    params['m2'] = m2
    params['c3'] = c3
    params['c2'] = c2

    save_dict(params_filename,params)

    
