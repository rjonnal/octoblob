from octoblob.data_source import DataSource
import octoblob.functions as blobf
import logging
from matplotlib import pyplot as plt
from octoblob import diagnostics

example_data_1_filename = '../data/test_1.unp'
d = diagnostics.Diagnostics(example_data_1_filename)

if __name__=='__main__':
    #df = DataSource(example_data_1_filename)
    df = blobf.get_source(example_data_1_filename)
    frame = df.next_frame()
    frame = blobf.fbg_align(frame,diagnostics=d)
    
    
