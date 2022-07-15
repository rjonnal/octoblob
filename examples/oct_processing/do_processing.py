import octoblob as blob
import multiprocessing as mp
import glob,sys
from matplotlib import pyplot as plt

files = glob.glob('./*.unp')

for fn in files:
    blob.processors.setup_processing(fn)
    blob.processors.optimize_mapping_dispersion(fn,mode='brightness',diagnostics=True)
    blob.processors.process_bscans(fn,diagnostics=False)
    plt.close('all')
