import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from octoblob.volume_tools import SyntheticVolume
from octoblob.ticktock import tick, tock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


sv = SyntheticVolume(120,50,150)

#sv.save_volumes('synthetic/synthetic',10,diagnostics=False)
sv.save_volumes('synthetic_rigid/synthetic',2,diagnostics=False,volume_rigid=True)
#sv.save_volumes('synthetic/synthetic',10,diagnostics=False,volume_rigid=False)
plt.figure()
sv.plot_history()
plt.show()
