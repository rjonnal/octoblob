import numpy as np

bit_shift_right = 4
dtype=np.uint16

# set fbg_position to None to skip fbg alignment
fbg_position = None
spectrum_start = 159
spectrum_end = 1459

fft_oversampling_size = None

# cropping parameters: negative numbers mean counting from the end of the array:
bscan_z1 = 1000
bscan_z2 = 1200

bscan_x1 = 0
bscan_x2 = -20

mapping_coefficients = [0.0e-10,0.0e-6,0.0,0.0]
dispersion_coefficients = [7.2e-09, -7.2e-05, 0.0, 0.0]

# parameters for bulk motion correction and phase variance calculation:
# original values:
# bulk_correction_threshold = 0.3
# phase_variance_threshold = 0.43

bulk_correction_threshold = 0.5
phase_variance_threshold = 0.5

# change this to false if it starts causing problems, but it should be stable:
use_multiprocessing = True

n_skip = 0


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


