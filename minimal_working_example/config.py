# general parameters
require_multiprocessing = True

# parameters for dispersion compensation
dB_clims = (40,90)
dc3_abs_max = 1e-7
dc2_abs_max = 1e-4
noise_roi = [10,100,10,100]
dispersion_frame_index = 50

# parameters for bscan generation
use_generic_mapping_dispersion_file = True
start_bscan = 80
end_bscan = 130
show_bscans = False
save_bscan_pngs = False
left_crop = 10
right_crop = 25
autocrop = True
# autocrop works by looking for the first and last pixels that are higher than some multiple
# of the image min. These factors are inner_threshold_factor and outer_threshold_factor (for
# the inner and outer edges of the retina, with some additional padding:
inner_threshold_factor = 2.0
outer_threshold_factor = 5.0
autocrop_padding = 20

# parameters for org preprocessing
phase_velocity_png_contrast_percentiles = (5,95)
amplitude_png_contrast_percentiles = (40,99)
variance_png_contrast_percentiles = (40,99)
residual_error_png_contrast_percentiles = (5,95)

block_size = 5 # number of B-scans to use in phase velocity estimation
bscan_interval = 2.5e-3 # time between B-scans
reference_bscan_filename = 'complex_00100.npy'

# parameters shifting histogram method
n_base_bins = 8
n_bin_shifts = 12
histogram_threshold_fraction = 0.05

write_pngs = True


