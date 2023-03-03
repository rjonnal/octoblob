# Basic B-scan processing

## Folder contents

* process.py: OCT/ORG processing script

* reset.sh: a bash script for deleting all of the downloaded and processed data, mainly for cleaning up this folder before pushing changes to github

## Download test data

To run this example you must download the test data from the links below:

* test.unp: the spectral data stored in raw binary 16 bit unsigned integer format. 

  > Download [test.unp](https://www.dropbox.com/s/pf6b951mlntqq9l/test.unp?dl=1)
.

* test.xml: acquisition parameters stored by the OCT instrumetation software during acquisition. 

  > Download [test.xml](https://www.dropbox.com/s/ux5qlinqq6y1zy4/test.xml?dl=1).

After downloading, put them into the `examples/generating_bscans` folder.


## B-scan processing

1. (Optional) Edit the file `process.py`, and edit the value assigned to `data_filename`.

2. Using the Anaconda terminal (command prompt), change into the `octoblob/examples/process.py` folder and run the program by issuing `python process.py` at the command prompt. If you've skipped step 1, you'll need to specify the `.unp` filename at the command prompt, e.g., `python process.py test.unp`.
