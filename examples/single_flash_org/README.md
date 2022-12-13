# Basic single flash ORG processing

## Folder contents

* process.py: OCT/ORG processing script

* plot_velocities.py: an interactive tool for visualizing outer segment phase changes

* reset.sh: a bash script for deleting all of the downloaded and processed data, mainly for cleaning up this folder before pushing changes to github

## Download test data

To run this example you must download the test data from the links below:

* test.unp: the spectral data stored in raw binary 16 bit unsigned integer format. 

  > Download [test.unp](https://www.dropbox.com/s/pf6b951mlntqq9l/test.unp?dl=1)
.

* test.xml: acquisition parameters stored by the OCT instrumetation software during acquisition. 

  > Download [test.xml](https://www.dropbox.com/s/ux5qlinqq6y1zy4/test.xml?dl=1).

After downloading, put them into the `examples/single_flash_org` folder.


## OCT/ORG processing

1. Edit the file `process.py`, and edit the values assigned to `data_filename`, `org_start_frame`, and `org_end_frame` as needed. For single flash experiments, only a subset of B-scans must be processed; see the code comment for details. For flicker experiments, the entire set of B-scans must be processed.

2. Run the program by issuing `python process.py` at the command prompt. This will take a few minutes. The ORG processing in particular is somewhat slow.

## ORG visualization

1. Run the program `plot_velocities.py` by issuing `python plot_velocities.py` at the command prompt. If run this way, the program searches recursively for folders called `org` in the current directory and its subdirectories. Alternatively, you may issue `python plot_velocities.py ./test_bscans` to search only that subdirectory (recursively). In these cases, the program will run on each of the `org` folders it finds. Finally, you may specify a particular org folder with `python plot_velocities.py ./test_bscans/org`, in which case it will run only on that folder.

2. The input required by the user is clicking the upper left and lower right corners of a rectangle containing just the IS/OS and COST bands to be analyzed, in the B-scan on the left. Within this rectangle, the bands should be approximately equidistant, to facilitate a simple segmentation algorithm. Selection of a rectangle causes the $v_{OS}$ plot for that region to appear in the right panel. When multiple rectangles are created, multiple plots are generated on the right, with the rectangles and plot lines color-coordinated for comparison. The `backspace` key deletes the last rectangle, and clicking outside of the B-scan on the left clears all of the rectangles. The `enter` key saves the figure and associated data in two places: the working directory, in a folder called 'plot_velocities_results' and in the `org` folder containing the raw ORG data.
