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

**TBD**
