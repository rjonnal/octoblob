import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
import scipy.ndimage as spn
import scipy.interpolate as spi
import imageio

def f(x):
    print(x+1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

import multiprocessing as mp

data = np.random.rand(10)
with mp.Pool(4) as p:
    print(p)
    p.map(f,data)
