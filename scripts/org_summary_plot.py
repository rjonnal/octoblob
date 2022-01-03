import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys

df_all =  pd.read_csv('master_summary.csv')

subject_array = [1,2,3]
subject_marker_array = ['s','o','x']
eccentricity_array = [2,4,6,8]


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

        plt.plot(eccentricity+x_offset,np.mean(vmax_arr),color_marker,label=label)
        plt.errorbar(eccentricity+x_offset,np.mean(vmax_arr),yerr=np.std(vmax_arr),ecolor='k',capsize=4)


        
plt.legend()
plt.show()
