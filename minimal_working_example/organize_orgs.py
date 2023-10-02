import os,glob,sys
from pathlib import Path
import numpy as np

magnitude_dictionary = {}

def read_looky(fn):
    with open(fn,'r') as fid:
        lines = fid.readlines()
    line = [l for l in lines if l.find('abs location')>-1][0]
    coords = line[line.find('(')+1:line.find(')')]
    coords = [float(d) for d in coords.split(',')]
    coords = np.array(coords)
    magnitude = np.sqrt(np.sum(coords**2))
    return coords,magnitude

for path in Path('.').rglob('org.txt'):
    pathstr = str(path)
    idx = pathstr.find('_bscans')
    assert idx>-1
    timestamp = pathstr[idx-8:idx]
    for lpath in Path('.').rglob('%s.looky'%timestamp):
        looky_fn = lpath
        coords,magnitude = read_looky(looky_fn)
        try:
            magnitude_dictionary[magnitude].append(path)
        except KeyError:
            magnitude_dictionary[magnitude] = [path]


for k in sorted(magnitude_dictionary.keys()):
    print('%0.2f deg sets:'%k)
    datasets = magnitude_dictionary[k]
    for d in datasets:
        print(d)
    print()
