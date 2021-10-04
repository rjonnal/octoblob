import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import time
import logging
from octoblob.volume_tools import SyntheticVolume, make_simple_volume_series
from octoblob.ticktock import tick, tock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


#make_simple_volume_series('simple')
#sys.exit()

sv = SyntheticVolume(120,50,150,rpower=1000,sphere_diameter=7,diagnostics=True)
#svreg = SyntheticVolume(120,50,150,rpower=1000,sphere_diameter=3,regular=True)


#sv_small = SyntheticVolume(12,10,8,sphere_diameter=7,motion=(3.0/120,2.0/50,1.0/150))
#sv_small.save_volumes('synthetic_rigid_small/synthetic',10,diagnostics=False,volume_rigid=True)
#sv.save_volumes('synthetic/synthetic',10,diagnostics=False)
#sv.save_volumes('synthetic_rigid/synthetic',5,diagnostics=False,volume_rigid=True)

sv.save_volumes('synthetic/synthetic',5,diagnostics=False,volume_rigid=False,motion_factor=1.5)
#svreg.save_volumes('synthetic_reg/synthetic',5,diagnostics=False,volume_rigid=False,motion_factor=1.5)

plt.show()
