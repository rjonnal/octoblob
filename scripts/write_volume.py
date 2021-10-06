import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from octoblob.volume_tools import Volume



# If the python command has less than 2 arguments, print the instructions and call it quits.
if len(sys.argv)<2:
    print('Usage: python test_project_enface.py input_directory')
    sys.exit()

# The second argument you specify in the command line is the input directory.
input_directory = sys.argv[1]

#  Create subfolder to the input directory to save the flattened data and not overwrite data.
output_directory = input_directory.strip('/').strip('\\')+'_tiff'

v = Volume(input_directory)
v.write_tiffs(output_directory)

