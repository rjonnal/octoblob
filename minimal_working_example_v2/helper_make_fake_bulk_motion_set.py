import numpy as np
import matplotlib.pyplot as plt
import sys,os,glob
import functions as blobf
from matplotlib.widgets import Button, Slider

try:
    input_filename = sys.argv[1]
except:
    print('Please supply the source bscan filename at the command line, i.e., the one to base the fake ones on.')
    sys.exit()


bscan = np.load(input_filename)

outdir = 'fake_bulk_motion_bscans'

os.makedirs(outdir,exist_ok=True)

out_indices = range(99,109)

dtheta = np.pi/12.0

for out_index in out_indices:
    fn = os.path.join(outdir,'complex_%05d.npy'%out_index)
    noise = np.random.randn(bscan.shape[0],bscan.shape[1])*np.pi/64
    bscan = bscan * np.exp(1j*dtheta)
    bscan = bscan * np.exp(1j*noise)
    print('Writing %s.'%fn)
    np.save(fn,bscan)
