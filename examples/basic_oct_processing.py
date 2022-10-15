from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics_tools
from octoblob import parameters
import sys

example_data_1_filename = 'test_1.unp'

#diagnostics = diagnostics_tools.Diagnostics(example_data_1_filename)
diagnostics = None

params = parameters.Parameters(example_data_1_filename)

if __name__=='__main__':
    #src = DataSource(example_data_1_filename)
    src = blobf.get_source(example_data_1_filename)
    
    spectra = src.next_frame(diagnostics=diagnostics)
    spectra = blobf.fbg_align(spectra,diagnostics=diagnostics)
    spectra = blobf.dc_subtract(spectra,diagnostics=diagnostics)
    spectra = spectra[k_crop_1:k_crop_2]

    
    
    print(spectra.shape)
    plt.imshow(spectra,aspect='auto')
    plt.show()
    sys.exit()
