import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys


edf = pd.read_csv('experimental_parameters_log.csv')
adf = pd.read_csv('analysis_parameters_log.csv')

# the analysis parameters log contains the following figures of merit; for any run of this program, we must specify
# which figures of merit to plot; a separate figure will be created for each figure of merit
# 'vmin_0_20': the minimum value between 0 and 20 ms
# 'vmax_20_40': the maximum value between 20 and 40 ms
# 'vmean_20_40': the mean value between 20 and 40 ms
# 'amin_0_50': the minimum (most negative) change in velocity (acceleration) between 0 and 50 ms
# 'amax_0_50': the maximum (most positive) change in velocity (acceleration) between 0 and 50 ms
# 'std_0_50': the standard deviation of the response between 0 and 50 ms
# 'mad_0_50': the mean absolute deviation of the response between 0 and 50 ms
figures_of_merit = ['vmin_0_20','vmean_20_40','amax_0_50']
eccentricities_to_omit = []
x_offset_fraction = 0.02


df_list = []

ecols = edf.columns
acols = adf.columns

for aidx,arow in adf.iterrows():
    adate = arow['date']
    atime = arow['time']

    d = {}

    
    d['date'] = adate
    d['time'] = atime

    for eidx,erow in edf.iterrows():
        edate = erow['date']
        etime = erow['time']

        if edate==adate and etime==atime:
            for ecol in ecols:
                d[ecol] = erow[ecol]
            for acol in acols:
                d[acol] = arow[acol]

    df_list.append(d)

df_all = pd.DataFrame(df_list)



label_dict = {}
label_dict['vmin_0_20'] = '$v_{min}$'
label_dict['vmax_20_40'] = '$v_{max}$'
label_dict['vmean_20_40'] = '$\overline{v_{20,40}}$'
label_dict['amin_0_50'] = '$(\Delta v)_{min}$'
label_dict['amax_0_50'] = '$(\Delta v)_{max}$'
label_dict['std_0_50'] = '$\sigma_{0,50}$'
label_dict['mad_0_50'] = '$\overline{|v|_{0,50}}$'

subject_array = list(df_all['subject'].unique())

print(subject_array)
subject_marker_array = ['s','o','x']
eccentricity_array = list(df_all['eccentricity'].unique())
bleaching_array = list(df_all['bleaching'].unique())
eccentricity_marker_array = ['s','o','x','^']

ecc_range = np.max(eccentricity_array)-np.min(eccentricity_array)
ecc_offset_factor = ecc_range*x_offset_fraction

b_range = np.max(bleaching_array)-np.min(bleaching_array)
b_offset_factor = b_range*x_offset_fraction

def err(vec):
    return np.std(vec)/np.sqrt(len(vec))

# first, let's look at eccentricity-dependence, with only 66% bleaching
df = df_all[df_all['bleaching']==66]

for idx,subject in enumerate(subject_array):
    subject_df = df[df['subject']==subject]
    subject_marker = subject_marker_array[idx]
    color_marker = 'k'+subject_marker

    x_offset = (idx-1)*ecc_offset_factor

    for eidx,eccentricity in enumerate(eccentricity_array):

        if eidx==0:
            label = 'subject %d'%subject
        else:
            label = None

        ecc_df = subject_df[subject_df['eccentricity']==eccentricity]

        for fidx,fom in enumerate(figures_of_merit):

            y_arr = np.array(ecc_df[fom])

            plt.figure(fidx+1)
            plt.plot(eccentricity+x_offset,np.mean(y_arr),color_marker,label=label)
            plt.errorbar(eccentricity+x_offset,np.mean(y_arr),yerr=err(y_arr),ecolor='k',capsize=4)
            plt.xlabel('eccentricity (deg)')
            plt.ylabel(label_dict[fom])

            
# done with df, so delete and reuse

df = df_all[df_all['subject']==1]

for idx,eccentricity in enumerate(eccentricity_array):
    ecc_df = df[df['eccentricity']==eccentricity]

    x_offset = (idx-1)*b_offset_factor

    if eccentricity in eccentricities_to_omit:
        continue
    
    for bidx,bleaching in enumerate(bleaching_array):
        
        if bidx==0:
            label = 'ecc %d'%eccentricity
        else:
            label = None

        
        bleach_df = ecc_df[ecc_df['bleaching']==bleaching]
        ecc_marker = eccentricity_marker_array[idx]

        for fidx,fom in enumerate(figures_of_merit):
            y_arr = np.array(bleach_df[fom])

            plt.figure(len(figures_of_merit)+fidx+1)
            plt.plot(bleaching+x_offset,np.mean(y_arr),'k'+ecc_marker,label=label)
            plt.errorbar(bleaching+x_offset,np.mean(y_arr),yerr=err(y_arr),ecolor='k',capsize=4)
            plt.xlabel('bleaching %')
            plt.ylabel(label_dict[fom])

for f in range(1,2*len(figures_of_merit)+1):
    plt.figure(f)
    plt.legend()
        
plt.show()
