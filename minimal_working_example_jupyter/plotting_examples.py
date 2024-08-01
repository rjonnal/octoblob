import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys
import pandas as pd
import functions as blobf

# Illustrate the use of the two configuration files
# Made-up data:
x = np.arange(100)
y = np.random.randn(100)

import plot_configuration_presentation as pcfg
pcfg.setup()

fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(x,y,marker='s',color=pcfg.colors[0],linestyle='')
pcfg.save('figures/fake_data_for_presentation.png')

import plot_configuration_manuscript as pcfg
pcfg.setup()

fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(x,y,marker='s',color=pcfg.colors[0],linestyle='')
pcfg.save('figures/fake_data_for_manuscript.png')

# Now let's go back to presentation configuration:
import plot_configuration_presentation as pcfg
pcfg.setup()


# New function: blobf.read_looky
ecc = blobf.read_looky('16_24_48')
# this gives a dictionary with easy programmatic access
# to looky/fixation info
print('output from functions.read_looky:')
print(ecc)
print(ecc['radius'])
print(ecc['horizontal_direction'])

# Example of plotting a single ORG
dat = np.loadtxt('16_24_48_bscans/org/org_000.txt')
t = dat[:,0]
org = dat[:,1]
fig = plt.figure()
ax = fig.subplots(1,1)
ax.plot(t,org,marker='none',linestyle='-',color=pcfg.colors[0])
pcfg.save('figures/sample_single_org.png')

# Example of plotting two ORGs as subplots:
# Example of plotting a single ORG
dat_01 = np.loadtxt('16_24_48_bscans/org/org_000.txt')
dat_02 = np.loadtxt('16_53_25_bscans/org/org_000.txt')
t = dat_01[:,0]
org_01 = dat_01[:,1]
org_02 = dat_02[:,1]

fig = plt.figure()
ax1,ax2 = fig.subplots(1,2)
ax1.plot(t,org_01,marker='none',linestyle='-',color=pcfg.colors[0])
ax1.set_xlabel('time (s)')
ax1.set_ylabel('velocity ($\mu m/s$)')

ax2.plot(t,org_02,marker='none',linestyle='-',color=pcfg.colors[0])
ax2.set_xlabel('time (s)')
# Omit ylabel on right plot:
# ax2.set_ylabel('velocity ($\mu m/s$)')

# better, more compact approach for setting labels:
for ax in [ax1,ax2]:
    ax.set_xlabel('time (s)')


pcfg.save('figures/sample_two_orgs_subplots.png')

