# Basic ORG processing for Optopol data

This example illustrates ORG processing for Optopol data that have already been processed into OCT amplitude and phase, and stored in `.bin` files.

## Folder contents

* process.py: OCT/ORG processing script

* plot_general_org.py: an interactive tool for visualizing phase changes between arbitrary, user-selected layers

* reset.sh: a bash script for deleting all of the downloaded and processed data, mainly for cleaning up this folder before pushing changes to github

## Download test data

To run this example you must download the test data from the links below:

[amplitude bin file](https://www.dropbox.com/s/efpieltzhry23nn/GB_1905021_GB_TrackingON_20230217101223_2_101x250x2048x2_Amp.bin?dl=1)

[phase bin file](https://www.dropbox.com/s/6xprkbeg8iff0xb/GB_1905021_GB_TrackingON_20230217101223_2_101x250x2048x2_Phase.bin?dl=1)

After downloading, put them into the `examples/optopol_interface` folder.

## OCT/ORG processing

1. Edit the file `process.py`, and edit the values assigned to `org_start_frame`, `org_end_frame` as needed. For single flash experiments, only a subset of B-scans must be processed; see the code comment for details. 

2. Using the Anaconda terminal (command prompt), change into the `octoblob/examples/optopol_interface` folder and run the program by issuing `python process.py FILENAME` at the command prompt. `FILENAME` should be replaced by the name of the file containing **amplitude** data, e.g., `GB_1905021_GB_TrackingON_20230217101223_2_101x250x2048x2_Amp.bin`. This will take a few minutes. The ORG processing in particular is somewhat slow.

## ORG visualization

0. If necessary, edit the following parameters in `plot_general_org.py`:

```python
stimulus_index = 50
bscan_rate = 400.0
zlim = (400,600) # depth limits for profile plot in um
vlim = (-5,5) # velocity limits for plotting in um/s
z_um_per_pixel = 3.0
```

1. Run the program `plot_general_org.py` by issuing `python plot_general_org.py` at the command prompt, in the same folder. If run this way, the program searches recursively for folders called `org` in the current directory and its subdirectories. This folder will by default be a subfolder of the `FILENAME_bscans` folder.

2. The input required by the user is clicking the end points of two line segments, one at a time. These line segments determine the layers between which phase velocities are computed. The user must click these line segments in a particular order--the left end of the top line segment, the right end of the top line segment, the left end of the bottom line segment, and the right end of the bottom line segment. The program will attempt to convert these line segments into arbitrary paths tracing the contour of the underlying layer by using the `refine_z` parameter:

```python
# refine_z specifies the number of pixels (+/-) over which the
# program may search to identify a local peak. The program begins by asking
# the user to trace line segments through two layers of interest. These layers
# may not be smooth. From one A-scan to the next, the brightest pixel or "peak"
# corresponding to the layer may be displaced axially from the intersection
# of the line segment with the A-scan. refine_z specifies the distance (in either
# direction, above or below that intersection) where the program may search for a
# brighter pixel with which to compute the phase. The optimal setting here will
# largely be determined by how isolated the layer of interest is. For a relatively
# isolated layer, such as IS/OS near the fovea, a large value may be best. For
# closely packed layers such as COST and RPE, smaller values may be useful. The
# user receives immediate feedback from the program's selection of bright pixels
# and can observe whether refine_z is too high (i.e., causing the wrong layer
# to be segmented) or too low (i.e., missing the brightest pixels.
```

Selection of these line segments causes the $v$ plot for that region to appear in the right panel. When multiple regions are created, multiple plots are generated on the right, with the rectangles and plot lines color-coordinated for comparison. The `backspace` key deletes the last region, and clicking outside of the B-scan on the left clears all of the regions. The `enter` key saves the figure and associated data in two places: the working directory, in a folder called `layer_velocities_results` in the `org` folder containing the raw ORG data.

## Example results

### Cone outer segment ORG responses

![Cone outer segment ORG responses](./figs/cone_os_org_optopol.png)

