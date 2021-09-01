### Instructions for using the simplified octoblob scripts

In general, these scripts should be copied to the folder containing the data files (or the parent folder, if data with identical parameters are organized in subfolders. Scripts are invoked as follows:

```python script_name.py path/to/data/file.unp flag1 flag2```

The ```octoblob/scripts``` folder contains an example dataset in ```octoblob/scripts/data```, which will be used for the examples below.

#### Initial ```parameters.py``` file

You must start with a ```parameters.py``` file with at least the following parameters defined:

```python
bit_shift_right = 4
dtype=np.uint16
fft_oversampling_size = 4096
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
```

#### Adjusting parameters in ```parameters.py```

Issue:

```python set_parameters.py data/oct_test_set.unp```

The resulting plots will assist you to adjust the following parameters; your exact values may differ slightly.

```python
fbg_position = None
spectrum_start = 159
spectrum_end = 1459

# cropping parameters: negative numbers mean counting from the end of the array:
bscan_z1 = 3100
bscan_z2 = -100
bscan_x1 = 0
bscan_x2 = -20
```

#### Estimating mapping and dispersion coefficients

Issue:

```python estimate_mapping_and_dispersion.py data/oct_test_set.unp```

The resulting interactive plots will allow you to set first the mapping and then the dispersion coefficients. Your last click determines the values that are printed to the terminal, and these should be copied and pasted into ```parameters.py```, e.g.:

```python
mapping_coefficients = [0.0,0.0,0.0,0.0]
dispersion_coefficients = [7.2e-09, -7.2e-05, 0.0, 0.0]
```

We sometimes use zeros for mapping coefficients even if other values make the image look slightly better, for simplicity. Non-zero mapping coefficients can generate artifacts in some cases.

#### Processing the data

To silently generate complex-valued ```.npy``` B-scan files:

```python process_bscans.py data/oct_test_set.unp data/oct_test_set.unp```

To generate complex-valued ```.npy``` B-scan files and PNG files:

```python process_bscans.py data/oct_test_set.unp data/oct_test_set.unp show```

To generate complex-valued ```.npy``` B-scan files and diagnostics:

```python process_bscans.py data/oct_test_set.unp data/oct_test_set.unp diagnostics```

To generate complex-valued ```.npy``` B-scan files, PNG files, and diagnostics:

```python process_bscans.py data/oct_test_set.unp data/oct_test_set.unp diagnostics show```


