import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys

df_all =  pd.read_csv('master_summary.csv')

subject_array = [1,2,3]
subject_marker_array = ['s','o','x']
eccentricity_array = [2,4,6,8]
bleaching_array = [8,17,33,66]
eccentricity_marker_array = ['s','o','x','^']


def err(vec):
    return np.std(vec)/np.sqrt(len(vec))

if True:
    # first, let's look at eccentricity-dependence, with only 66% bleaching
    df = df_all[df_all['bleaching']==66]

    for idx,subject in enumerate(subject_array):
        subject_df = df[df['subject']==subject]
        subject_marker = subject_marker_array[idx]
        color_marker = 'k'+subject_marker

        x_offset = (idx-1)*0.1

        for eccentricity in eccentricity_array:

            if eccentricity==2:
                label = 'subject %d'%subject
            else:
                label = None

            ecc_df = subject_df[subject_df['eccentricity']==eccentricity]

            vmin_arr = np.array(ecc_df['velocity min'])
            vmax_arr = np.array(ecc_df['velocity max'])
            plt.figure(1)
            plt.plot(eccentricity+x_offset,np.mean(vmin_arr),color_marker,label=label)
            plt.errorbar(eccentricity+x_offset,np.mean(vmin_arr),yerr=err(vmin_arr),ecolor='k',capsize=4)
            plt.xlabel('eccentricity (deg)')
            plt.ylabel('$v_{min}$')
            
            plt.figure(2)
            plt.plot(eccentricity+x_offset,np.mean(vmax_arr),color_marker,label=label)
            plt.errorbar(eccentricity+x_offset,np.mean(vmax_arr),yerr=err(vmax_arr),ecolor='k',capsize=4)
            plt.xlabel('eccentricity (deg)')
            plt.ylabel('$v_{max}$')

plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()

# done with df, so delete and reuse

df = df_all[df_all['subject']==1]

for idx,eccentricity in enumerate(eccentricity_array):
    ecc_df = df[df['eccentricity']==eccentricity]

    x_offset = (idx-1)

    if eccentricity in [8]:
        continue
    
    for bleaching in bleaching_array:
        
        if bleaching==8:
            label = 'ecc %d'%eccentricity
        else:
            label = None
            
        bleach_df = ecc_df[ecc_df['bleaching']==bleaching]
        vmin_arr = np.array(bleach_df['velocity min'])
        vmax_arr = np.array(bleach_df['velocity max'])
        ecc_marker = eccentricity_marker_array[idx]

        plt.figure(3)
        plt.plot(bleaching+x_offset,np.mean(vmin_arr),'k'+ecc_marker,label=label)
        plt.errorbar(bleaching+x_offset,np.mean(vmin_arr),yerr=err(vmin_arr),ecolor='k',capsize=4)
        plt.xlabel('bleaching %')
        plt.ylabel('$v_{min}$')
        
        plt.figure(4)
        plt.plot(bleaching+x_offset,np.mean(vmax_arr),'k'+ecc_marker,label=label)
        plt.errorbar(bleaching+x_offset,np.mean(vmax_arr),yerr=err(vmax_arr),ecolor='k',capsize=4)
        plt.xlabel('bleaching %')
        plt.ylabel('$v_{max}$')
        
plt.figure(3)
plt.legend()
plt.figure(4)
plt.legend()
plt.show()
