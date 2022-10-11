from octoblob.system_label import system_label

if system_label=='clinical_org':
    default_parameter_dictionary = {}
    default_parameter_dictionary['bit_shift_right'] = 4
    default_parameter_dictionary['dtype'] = 'u2'

    # now obsolete:
    #default_parameter_dictionary['fbg_position'] = 100
    #default_parameter_dictionary['fbg_region_height'] = 30
    default_parameter_dictionary['fbg_noise_pixels'] = 80
    default_parameter_dictionary['fbg_search_distance'] = 11
    
    default_parameter_dictionary['spectrum_start'] = 159
    default_parameter_dictionary['spectrum_end'] = 1459
    default_parameter_dictionary['bscan_z1'] = 1000
    default_parameter_dictionary['bscan_z2'] = 1300
    default_parameter_dictionary['bscan_x1'] = 0
    default_parameter_dictionary['bscan_x2'] = 250
    default_parameter_dictionary['n_skip'] = 0
    default_parameter_dictionary['fft_oversampling_size'] = -1

    # now obsolete due to automation:
    #default_parameter_dictionary['eye'] = 'RE'
    #default_parameter_dictionary['ecc_horizontal'] = 0.0
    #default_parameter_dictionary['ecc_vertical'] = 0.0
    #default_parameter_dictionary['notes'] = ''

    # RSJ, 29 Sep 2022
    # changing dispersion compensation expression approach
    # from listing parameters 'c3','c2', etc. and explicit
    # 3rd order, to implicit order depending on number of coefficients.
    # Thus, each of the following lists must have the same numbers of items.
    # If they have 2 items, that implies 3rd order, since the 1st and 0th
    # terms are not used.
    # More explicitly, if values [c3,c2] are entered here, the optimization
    # will work on the polynomial specified by [c3,c2,0.0,0.0], since
    # neither the linear nor the constant terms have any affect on image
    # quality, either in mapping or dispersion compensation.
    # The order of the dispersion and mapping steps need not be the same.
    # For instance, dispersion could operate in 3rd order [c3,c2,0.0,0.0]
    # and mapping in second order [m2,0.0,0.0].
    # default_parameter_dictionary['dispersion_initial'] = [0.0,0.0]
    # default_parameter_dictionary['cmax'] = [1e-8,1e-4]
    # default_parameter_dictionary['cmin'] = [-1e-8,-1e-4]

    default_parameter_dictionary['dispersion_initial'] = [0.0]
    default_parameter_dictionary['cmax'] = [1e-4]
    default_parameter_dictionary['cmin'] = [-1e-4]

    default_parameter_dictionary['mapping_initial'] = [0.0,0.0]
    default_parameter_dictionary['mmax'] = [1e-8,1e-5]
    default_parameter_dictionary['mmin'] = [-2e-7,-1e-5]
    

elif system_label=='eyepod':
    default_parameter_dictionary = {}
    default_parameter_dictionary['bit_shift_right'] = 4
    default_parameter_dictionary['dtype'] = 'u2'
    default_parameter_dictionary['fbg_position'] = -1
    default_parameter_dictionary['fbg_region_height'] = 0
    default_parameter_dictionary['spectrum_start'] = 45
    default_parameter_dictionary['spectrum_end'] = 1245
    default_parameter_dictionary['bscan_z1'] = 850
    default_parameter_dictionary['bscan_z2'] = 1150
    # NOTE: for the following two parameters, the B-scan is assumed
    # to be 1080 pixels in width. This width is due to the fact that
    # the eyepod software (in contrast to the clinical_org software)
    # stores 3 repeats in a single frame (as opposed to separate frames).
    # This issue can be addressed by modifying parameters in the acquisition
    # XML file--changing the Number_of_BM_scans from what it is (e.g., 3) to
    # 1, and dividing the Number_of_Frames by 3.
    # If you do this, you have to change the default value of bscan_x2 below
    # to 360 instead of 1080.
    default_parameter_dictionary['bscan_x1'] = 0
    default_parameter_dictionary['bscan_x2'] = 1080
    default_parameter_dictionary['n_skip'] = 0
    default_parameter_dictionary['fft_oversampling_size'] = -1
    default_parameter_dictionary['eye'] = 'RE'
    default_parameter_dictionary['ecc_horizontal'] = 0.0
    default_parameter_dictionary['ecc_vertical'] = 0.0
    default_parameter_dictionary['notes'] = ''


    # RSJ, 29 Sep 2022
    # changing dispersion compensation expression approach
    # from listing parameters 'c3','c2', etc. and explicit
    # 3rd order, to implicit order depending on number of coefficients.
    # Thus, each of the following lists must have the same numbers of items.
    # If they have 2 items, that implies 3rd order, since the 1st and 0th
    # terms are not used.
    # More explicitly, if values [c3,c2] are entered here, the optimization
    # will work on the polynomial specified by [c3,c2,0.0,0.0], since
    # neither the linear nor the constant terms have any affect on image
    # quality, either in mapping or dispersion compensation.
    # The order of the dispersion and mapping steps need not be the same.
    # For instance, dispersion could operate in 3rd order [c3,c2,0.0,0.0]
    # and mapping in second order [m2,0.0,0.0].
    default_parameter_dictionary['dispersion_initial'] = [0.0,0.0]
    default_parameter_dictionary['cmax'] = [1e-8,1e-4]
    default_parameter_dictionary['cmin'] = [-1e-8,-1e-4]

    default_parameter_dictionary['mapping_initial'] = [0.0,0.0]
    default_parameter_dictionary['mmax'] = [1e-8,1e-5]
    default_parameter_dictionary['mmin'] = [-2e-7,-1e-5]
    

