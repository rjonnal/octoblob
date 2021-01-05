# This file contains parameters for the processing
# pipeline. It is meant to be frozen in releases of
# octoblob, such that processing can be reproduced
# perfectly without having to fiddle with them.

# coefficients for resampling lambda into k
# these coefficients c specify a polynomial p:
# p(x) = c_0*x^3 + c_1*x^2 + c_2*x + c_3
# p(x) is a the sampling error in x,
# and the measured spectra are interpolated from
# x -> x+p(x)
k_resampling_coefficients = [12.5e-10,-12.5e-7,0,0]

# these are the coefficients for the unit-amplitude
# phasor used for removing dispersion chirp; if the
# coefficients are c, then
# p(x) = c_0*x^3 + c_1*x^2 + c_2*x + c_3
# the dechirping phasor D is given by:
# D = e^[-i*p(x)]
# the spectra are dechirped by:
# dechirped_spectrum = spectra*D
dispersion_coefficients = [0.0,1.5e-6,0.0,0.0]

# the width of the window for gaussian windowing:
gaussian_window_sigma = 0.9

# paramters for bulk motion estimation, including
# smoothing by shifting bin edges; see Makita, 2006
# for a detailed description of the approach;
# in short, we do a histogram of the B-scan to B-scan
# phase error, with a fixed number of bins (n_bins);
# then, we shift the bin edges by a fraction of a
# bin width and recompute the histogram; the fractional
# shift is equal to 1/resample_factor
bulk_motion_n_bins = 16
bulk_motion_resample_factor = 24
bulk_motion_n_smooth = 5

# critical parameters: thresholds for bulk motion correction
# and phase variance computation
bulk_correction_threshold = 0.3
phase_variance_threshold = 0.43

