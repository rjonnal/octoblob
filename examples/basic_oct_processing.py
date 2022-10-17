from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics_tools
from octoblob import parameters
import sys,os

from octoblob import mapping_dispersion_optimizer as mdo

example_data_1_filename = 'test_1.unp'

#diagnostics = diagnostics_tools.Diagnostics(example_data_1_filename)
diagnostics = None

params_filename = os.path.join(os.path.split(example_data_1_filename)[0],'processing_parameters.json')

params = parameters.Parameters(params_filename)

k_crop_1 = 100
k_crop_2 = 1490





if __name__=='__main__':
    #src = DataSource(example_data_1_filename)
    src = blobf.get_source(example_data_1_filename)
    
    spectra = src.next_frame(diagnostics=diagnostics)
    spectra = blobf.fbg_align(spectra,diagnostics=diagnostics)
    spectra = blobf.dc_subtract(spectra,diagnostics=diagnostics)
    spectra = spectra[k_crop_1:k_crop_2,:]

    coefs = mdo.run(spectra,show=True)
    plt.show()
    sys.exit()
