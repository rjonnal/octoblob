import numpy as np

###############################################
# Critical parameters; must be set in all cases:
bit_shift_right = 4
dtype=np.uint16
fft_oversampling_size = None

# Parameter estimation limits
# To use the parameter estimation UI for dispersion compensation and
# mapping parameters, you have to set the limits for the coefficients
# in the display. If you are using the UIs and find either that the
# optimal parameters are very close to the center or very close to the
# edge of the window, these can be adjusted to modify the search space.
# In both dispersion and mapping, the third order coefficients are in the
# y-dimension in the window and the second order coefficients in the x.

# dispersion compensation limits
c3max = 1e-8
c3min = -1e-8
c2max = 1e-4
c2min = -1e-4

# mapping limits
m3max = 1e-8
m3min = -2e-7
m2max = 1e-5
m2min = -1e-5

# Leave n_skip at 0 unless you have orphaned angiography
# BM scans at the start of the file (FDML issue only)
n_skip = 0

# Leave this False for now
use_multiprocessing = False
##############################################


##############################################
# Spectrum parameters:
# set fbg_position to None to skip fbg alignment

fbg_position = 93
spectrum_start = 130
spectrum_end = 1460

##############################################


##############################################
# B-scan cropping parameters:
# cropping parameters: negative numbers mean counting from the end of the array:

bscan_z1 = 1050
bscan_z2 = 1300
bscan_x1 = 0
bscan_x2 = 250

##############################################


##############################################
# Processing coefficients
# mapping_coefficients = [0.0e-10,0.0e-6,0.0,0.0]
# dispersion_coefficients = [7.2e-09, -7.2e-05, 0.0, 0.0]

# From optimizer:
mapping_coefficients = [-1.3e-08, 5.4e-08, 0.0, 0.0]
dispersion_coefficients = [-1.2e-10, 2.9e-06, 0.0, 0.0]


#Last clicked mapping coefficients:
# mapping_coefficients = [-2.0e-08, 3.0e-07, 0.0, 0.0]
#Last clicked dispersion coefficients:
# dispersion_coefficients = [-1.3e-09, -3.1e-05, 0.0, 0.0]

##############################################


##############################################
# Angiography parameters
# parameters for bulk motion correction and phase variance calculation:
# original values:

#bulk_correction_threshold = 0.3
#phase_variance_threshold = 0.43

# new, exploratory values:

#bulk_correction_threshold = 0.5
#phase_variance_threshold = 0.5
##############################################

##############################################
# Optional display parameters:

png_aspect_ratio = 0.5
png_dB_clim = (40,90)
##############################################





