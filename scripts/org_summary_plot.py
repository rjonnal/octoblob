import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import octoblob.plotting_functions as opf



edf = pd.read_csv('experimental_parameters_log_testing.csv')
adf = pd.read_csv('analysis_parameters_log.csv')

bleaching_subject = 1

# the analysis parameters log contains the following figures of merit; for any run of this program, we must specify
# which figures of merit to plot; a separate figure will be created for each figure of merit
# 'vmin_0_20': the minimum value between 0 and 20 ms
# 'vmax_20_40': the maximum value between 20 and 40 ms
# 'vmean_20_40': the mean value between 20 and 40 ms
# 'amin_0_50': the minimum (most negative) change in velocity (acceleration) between 0 and 50 ms
# 'amax_0_50': the maximum (most positive) change in velocity (acceleration) between 0 and 50 ms
# 'std_0_50': the standard deviation of the response between 0 and 50 ms
# 'mad_0_50': the mean absolute deviation of the response between 0 and 50 ms
figures_of_merit = ['vmin_0_20','vmean_20_40','amax_0_50', 'amin_0_50']
eccentricities_to_omit = []
x_offset_fraction = 0.02
figure_size = (4,3) # w,h in inches
label_dict = {}
label_dict['vmin_0_20'] = '$v_{min}$ ($\mu m/s$)'
label_dict['vmax_20_40'] = '$v_{max}$ ($\mu m/s$)'
label_dict['vmean_20_40'] = '$\overline{v_{20,40}}$ ($\mu m/s$)'
label_dict['amin_0_50'] = '$(\Delta v)_{min}$ ($\mu m/s^2$)'
label_dict['amax_0_50'] = '$(\Delta v)_{max}$ ($\mu m/s^2$)'
label_dict['std_0_50'] = '$\sigma_{0,50}$ ($\mu m/s$)'
label_dict['mad_0_50'] = '$\overline{|v|_{0,50}}$ ($\mu m/s$)'

ylim_dict = {}
ylim_dict['vmin_0_20'] = [-3.5,0]
ylim_dict['vmax_20_40'] = [0,3]
ylim_dict['vmean_20_40'] = [0,2]
ylim_dict['amin_0_50'] = [-2.5,0]
ylim_dict['amax_0_50'] = [0,1.75]
ylim_dict['std_0_50'] = [0,3]
ylim_dict['mad_0_50'] = [0,3]

figure_size = (4,3) # (width_inches, height_inches)
font = 'Arial'
font_size = 10
screen_dpi = 100
print_dpi = 300
plot_linewidth = 2

# spines (lines at edges of plot)
spine_color = 'k'
spine_linewidth = 2

# legend appearance
show_legend_frame = True
legend_linewidth = 2
legend_edgecolor = 'k'
legend_facecolor = 'w'
legend_alpha = 1.0

# see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
#style = 'bmh'
style = 'ggplot'
#style = 'seaborn-deep'
#style = 'fivethirtyeight'
#############################################################################################################

opf.setup_plots(style=style,font_size=font_size,font=font)

for fig_idx in range(1,2*len(figures_of_merit)+1):
    plt.figure(figsize=figure_size)


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
df_all.to_csv('merged_parameters_log.csv')



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

def format_legend(leg):
    leg.get_frame().set_edgecolor(legend_edgecolor)
    leg.get_frame().set_facecolor(legend_facecolor)
    leg.get_frame().set_linewidth(legend_linewidth)
    leg.get_frame().set_alpha(legend_alpha)


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

df = df_all[df_all['subject']==bleaching_subject]
mult = 1
if len(df)>0:
    mult = 2
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


for f in range(1,mult*len(figures_of_merit)+1):
    fom = figures_of_merit[(f-1)%len(figures_of_merit)]
    try:
        plt.figure(f)
        leg = plt.legend()
        format_legend(leg)
        
        ax = plt.gca()
        ax.tick_params(direction='in')
        ax.tick_params(left=True)
        ax.tick_params(right=True)
        ax.tick_params(top=True)
        ax.tick_params(bottom=True)
        ax.set_ylim(ylim_dict[fom])

        for spine in ['top','bottom','left','right']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(spine_linewidth)

        plt.tight_layout()

        
        if f<=len(figures_of_merit):
            plt.savefig('ecc_dependence_%s.svg'%fom)
        else:
            plt.savefig('bleaching_dependence_%s.svg'%fom)
    except:
        pass
    
        
plt.show()
