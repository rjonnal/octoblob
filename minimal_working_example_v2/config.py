# general parameters
require_multiprocessing = True

# parameters for dispersion compensation
dB_clims = (40,90)
dc3_abs_max = 1e-7
dc2_abs_max = 1e-4
noise_roi = [10,100,10,100]
dispersion_frame_index = 50

# parameters for bscan generation
use_generic_mapping_dispersion_file = False
start_bscan = 80 # for ORG
end_bscan = 130 # for ORG
stimulus_index = 20 # if start_bscan is 80

show_bscans = False
save_bscan_pngs = False
left_crop = 10
right_crop = 25
autocrop = False
autocrop_stride = 2 # how frequently to load bscans for the sake of cropping, i.e. every nth scan

# autocrop works by looking for the first and last pixels that are higher than some multiple
# of the image min. These factors are inner_threshold_factor and outer_threshold_factor (for
# the inner and outer edges of the retina, with some additional padding:
inner_threshold_factor = 2.0
outer_threshold_factor = 5.0
autocrop_padding = 100
oversample = 2
k_crop_1 = 100
k_crop_2 = 1490

# parameters for org preprocessing
phase_velocity_png_contrast_percentiles = (5,95)
amplitude_png_contrast_percentiles = (40,99)
variance_png_contrast_percentiles = (40,99)
residual_error_png_contrast_percentiles = (5,95)
block_size = 3 # number of B-scans to use in phase velocity estimation
bscan_interval = 2.5e-3 # time between B-scans
# the reference bscan is used for aligning B-scans with one another axially, to flatten the series
reference_bscan_filename = 'complex_00500.npy'

# parameters for the shifting histogram method of bulk motion correction
n_base_bins = 8
n_bin_shifts = 12
histogram_threshold_fraction = 0.05

# save the preprocessed ORG data as PNG files too for ease of viewing
write_pngs = False


# OCTA parameters
phase_variance_png_contrast_percentiles = (5,95)
amplitude_variance_png_contrast_percentiles = (5,95)


