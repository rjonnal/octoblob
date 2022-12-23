# Basic flicker ORG processing

## Folder contents

* process.py: OCT/ORG processing script

* reset.sh: a bash script for deleting all of the downloaded and processed data, mainly for cleaning up this folder before pushing changes to github

## Download test data

To run this example you must download the test data from the links below:

* test_flicker.unp: the spectral data stored in raw binary 16 bit unsigned integer format. 

  > Download [test_flicker.unp](https://www.dropbox.com/s/fbms4ekrwvngt0a/test_flicker.unp?dl=0)
.

* test_flicker.xml: acquisition parameters stored by the OCT instrumetation software during acquisition. 

  > Download [test_flicker.xml](https://www.dropbox.com/s/hmny5xafcizj67q/test_flicker.xml?dl=0)

After downloading, put them into the `examples/flicker_org` folder.


## OCT/ORG processing

1. Using the Anaconda terminal (command prompt), change into the `octoblob/examples/flicker_org` folder and run the program by issuing `python process.py` at the command prompt. This may take 20-30 minutes. The ORG processing is slow, and 1600-2000 blocks must be processed.

## ORG visualization

1. Run the program `plot_velocities.py` by issuing `python plot_velocities.py` at the command prompt, in the same folder. If run this way, the program searches recursively for folders called `org` in the current directory and its subdirectories. Alternatively, you may issue `python plot_velocities.py ./test_bscans` to search only that subdirectory (recursively). In these cases, the program will run on each of the `org` folders it finds. Finally, you may specify a particular org folder with `python plot_velocities.py ./test_bscans/org`, in which case it will run only on that folder.

2. The input required by the user is clicking the upper left and lower right corners of a rectangle containing just the IS/OS and COST bands to be analyzed, in the B-scan on the left. Within this rectangle, the bands should be approximately equidistant, to facilitate a simple segmentation algorithm. Selection of a rectangle causes the $v_{OS}$ plot for that region to appear in the center panel and the power spectrum of $v_{OS}$ plotted in log scale in the right panel, limited to the range [0 Hz, 30Hz]. When multiple rectangles are created, multiple plots are generated on the right, with the rectangles and plot lines color-coordinated for comparison. The `backspace` key deletes the last rectangle, and clicking outside of the B-scan on the left clears all of the rectangles. The `enter` key saves the figure and associated data in two places: the working directory, in a folder called `plot_velocities_results` and in the `org` folder containing the raw ORG data.
