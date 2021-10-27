import glob,os,sys
import parameters as params
from process_bscans import process
import octoblob as blob

use_multiprocessing = params.use_multiprocessing
if use_multiprocessing:
    import multiprocessing as mp

args = sys.argv[1:]
args = blob.expand_wildcard_arguments(args)

files = []
flags = []

for arg in args:
    if os.path.exists(arg):
        files.append(arg)
    else:
        flags.append(arg.lower())

diagnostics = 'diagnostics' in flags
show_processed_data = 'show' in flags

files.sort()

try:
    n_workers = params.multiprocessing_n_processes
except Exception as e:
    n_workers = 4

def proc(f):
    process(f,diagnostics=diagnostics,show_processed_data=show_processed_data)
        
if __name__=='__main__':

    if use_multiprocessing:
        with mp.Pool(n_workers) as p:
            p.map(proc,files)

    else:
        for f in files:
            process(f,diagnostics=diagnostics,show_processed_data=show_processed_data)

