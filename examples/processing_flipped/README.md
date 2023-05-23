# Processing flipped data

Like the `examples/fbg_alignment` example, this example exposes more of the steps in OCT post-processing to the user, this time in order to permit 1) flipping of B-scans collected in the so-called "EDI" (extended depth imaging) mode. This simply means matching the reference arm length with the choroid, such that the outer retinal OCT is acquired with higher SNR than the inner retina. In our Axsun system, this results in the retina being inverted in the positive frequency DFT--the side of the DFT we ordinarily process. These B-scans must be flipped in order to do subsequent ORG processing.

The general goal of exposing more of the OCT post-processing to the user is to enable more flexibility. Here, instead of calling `octoblob.functions.spectra_to_bscan`, a local function `spectra_to_bscan` is called, which itself calls functions from the `octoblob.functions` library. Similar to `examples/fbg_alignment`, this version employs cross-correlation to align spectra to the FBG features. This approach appears, for the time being, to be as robust as the older method that aligned by most positive or most negative gradient, and has the additional benefit of not requiring the FBG features to be optimized (via the polarization controller just downstream of the source). As long as the FBG has some clear effect on the acquired spectra, they can be aligned and made phase stable by cross-correlation.

The flipping of the B-scan is done with a single line of code in the `spectra_to_bscan` function:

```python
# Flip the B-scan, since it was acquired near the top of the view, inverted
bscan = bscan[::-1,:]
```

Please see documentation on [extended slices](https://docs.python.org/release/2.3.5/whatsnew/section-slices.html) and the effect of negative strides for an explanation of how this works.

## Folder contents

* process.py: OCT/ORG processing script

* plot_general_org.py: ORG visualization (see `examples/single_flash_org_general` for more information)

* reset.sh: a bash script for deleting all of the downloaded and processed data, mainly for cleaning up this folder before pushing changes to github


## B-scan processing 

1. Using the Anaconda terminal (command prompt), change into the `octoblob/examples/processing_flipped_bscans` folder and run the program by issuing `python process.py XXX.unp`, where `XXX.unp` contains raw data from inverted B-scan acquisition.

2. By default, this script does ORG processing on the resulting B-scans as well.
