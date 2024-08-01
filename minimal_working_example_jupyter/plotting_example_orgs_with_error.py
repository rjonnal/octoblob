import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys
import pandas as pd
import functions as blobf

# Illustrate the use of the two configuration files
# Made-up data:

files = ['./16_24_48_bscans/org/org_000.txt',
         './16_53_25_bscans/org/org_000.txt']

# next is a list comprehension; very powerful syntax,
# read about it here:
# https://www.w3schools.com/python/python_lists_comprehension.asp
data = [np.loadtxt(f) for f in files]

# time is the same for all data, so just load it once:
t_arr = data[0][:,0]

# load the orgs into a list (another list comprehension):
org_list = [d[:,1] for d in data]

# convert org_list to a 2D array:
org_arr = np.array(org_list)

# it will be 2xN, so transpose; not necessary, but I
# like to keep the row index aligned with time
org_arr = org_arr.T

org_mean = np.mean(org_arr,axis=1)
org_std = np.std(org_arr,axis=1)


import plot_configuration_presentation as pcfg
pcfg.setup()

fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(t_arr,org_mean,marker='',color='k',linestyle='-')
ax.fill_between(t_arr,org_mean-org_std,org_mean+org_std,color=[0.5,0.5,0.5],alpha=0.5)

pcfg.save('figures/multple_orgs_with_error.png')
plt.show()
