import octoblob as blob
import multiprocessing as mp
import glob,sys,os,shutil
from matplotlib import pyplot as plt

files = glob.glob('./*.unp')
bscan_folders = [fn.replace('.unp','_bscans') for fn in files]


# for ORG processing we just use t = [-50,50] ms--that's 40 scans, from 20 to 60

for fn,bscan_folder in zip(files,bscan_folders):
    if not os.path.exists(bscan_folder):
        param_fn = blob.processors.get_param_filename(fn)
        blob.processors.setup_processing(fn,copy_existing=True)
        try:
            d = blob.processors.load_dict(param_fn)
            assert all([k in d.keys() for k in ['c2','c3','m2','m3']])
        except AssertionError:
            blob.processors.optimize_mapping_dispersion(fn,mode='brightness',diagnostics=True)
            
        blob.processors.process_bscans(fn,diagnostics=False,start_frame=20,end_frame=60)

