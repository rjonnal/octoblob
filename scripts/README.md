### Instructions for using the simplified octoblob scripts

In general, these scripts should be copied to the folder containing the data files (or the parent folder, if data with identical parameters are organized in subfolders. Scripts are invoked as follows:

```python script_name.py path/to/data/file.unp flag1 flag2```

The ```octoblob/scripts``` folder contains an example dataset in ```octoblob/scripts/data```, which will be used for the examples below.

#### Initial ```parameters.py``` file

You must start with a ```parameters.py``` file with at least the following parameters defined:

```python
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
```

#### Adjusting parameters in ```parameters.py```

Issue:

```python parameters_helper.py data/oct_test_set.unp```

The resulting plots will assist you to adjust the following parameters; your exact values may differ slightly.

```python
##############################################
# Spectrum parameters:
# set fbg_position to None to skip fbg alignment

fbg_position = None
spectrum_start = 159
spectrum_end = 1459

##############################################


##############################################
# B-scan cropping parameters:
# cropping parameters: negative numbers mean counting from the end of the array:

bscan_z1 = 1000
bscan_z2 = 1200
bscan_x1 = 0
bscan_x2 = -20

##############################################
```

#### Starting with fresh settings in ```parameters.py```

To start from scratch, first delete ```parameters.py``` and copy ```parameters_template.py``` to ```parameters.py```:

```cp parameters_template.py parameters.py```

Next, you will run ```parameters_helper.py``` twice. The first time will help you to enter values for ```fbg_position```, ```spectrum_start``` and ```spectrum_end```, and the second time will use these values to give a B-scan which will help you to set the B-scan cropping parameters.

Note: The ```fbg_position``` is relative to the top of the **uncropped** spectra. In other words, use the absolute position of the FBG trough in the first figure to note the approximate FBG location.

#### Estimating mapping and dispersion coefficients

Issue:

```python estimate_mapping_and_dispersion.py data/oct_test_set.unp```

The resulting interactive plots will allow you to set first the mapping and then the dispersion coefficients. Your last click determines the values that are printed to the terminal, and these should be copied and pasted into ```parameters.py```, e.g.:

```python
mapping_coefficients = [0.0,0.0,0.0,0.0]
dispersion_coefficients = [7.2e-09, -7.2e-05, 0.0, 0.0]
```

We sometimes use zeros for mapping coefficients even if other values make the image look slightly better, for simplicity. Non-zero mapping coefficients can generate artifacts in some cases.

#### Generating B-scans

To silently generate complex-valued ```.npy``` B-scan files:

```python process_bscans.py data/oct_test_set.unp```

To generate complex-valued ```.npy``` B-scan files and PNG files:

```python process_bscans.py data/oct_test_set.unp show```

To generate complex-valued ```.npy``` B-scan files and diagnostics:

```python process_bscans.py data/oct_test_set.unp diagnostics```

To generate complex-valued ```.npy``` B-scan files, PNG files, and diagnostics:

```python process_bscans.py data/oct_test_set.unp diagnostics show```


#### Phase analysis of B-scans

This script moves through the processed B-scans in overlapping blocks, e.g. frames 0-4, 1-5, 2-6, etc.. In each block, the frames are rigid-body registered, bulk-phase corrected to the first B-scan in the block, and then the phase change over the block is computed for each pixel. Two thresholds have to be set, for bulk-motion correction and computing the signal. Both are expressed as fractions of the B-scan's maximum amplitude. More pixels should be used for histogram-based bulk motion correction (i.e., higher threshold) than phase ramp computation.

Three parameters to adjust near the top of the script are:

```python
block_size = 5
histogram_threshold_fraction = 0.2
signal_threshold_fraction = 0.05
```
To run the phase analysis:

```python phase_analysis.py data/oct_test_set.unp```

