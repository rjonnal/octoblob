import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
import octoblob.plotting_functions as opf
from matplotlib.lines import Line2D


##############################################################################
bleaching_subject = 1
bleaching_plotting_function = plt.semilogx
# oct parameters
lambda_1 = 930e-9
lambda_2 = 1180e-9
n_samples = 1536

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
normalization_column = 'COST_ISOS_px'

label_dict = {}
label_dict['vmin_0_20'] = '$v_{min}$ ($\mu m/s$)'
label_dict['vmax_20_40'] = '$v_{max}$ ($\mu m/s$)'
label_dict['vmean_20_40'] = '$\overline{v}_{20,40}$ ($\mu m/s$)'
label_dict['amin_0_50'] = '$(\Delta v)_{min}$ ($\mu m/s^2$)'
label_dict['amax_0_50'] = '$(\Delta v)_{max}$ ($\mu m/s^2$)'
label_dict['std_0_50'] = '$\sigma_{0,50}$ ($\mu m/s$)'
label_dict['mad_0_50'] = '$\overline{|v|_{0,50}}$ ($\mu m/s$)'

autoscale = True
ylim_dict = {}
ylim_dict['vmin_0_20'] = [-3.5,0]
ylim_dict['vmax_20_40'] = [0,3]
ylim_dict['vmean_20_40'] = [0,2]
ylim_dict['amin_0_50'] = [-2.5,0]
ylim_dict['amax_0_50'] = [0,1.75]
ylim_dict['std_0_50'] = [0,3]
ylim_dict['mad_0_50'] = [0,3]

# edit filenames below as needed:
edf = pd.read_csv('experimental_parameters_log.csv')
adf = pd.read_csv('analysis_parameters_log.csv')

figure_size = (4,4) # (width_inches, height_inches)
ax_box = [.18,.15,.77,.80]
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

# how much to offset (jitter) markers for visibility:
x_offset_fraction = 0.02

# Determines whether to use automatic x-ticks or the values of x in the data:
ecc_xticks_actual = False
bleaching_xticks_actual = True

# plot markers
subject_marker_array = ['s','o','x']
eccentricity_marker_array = ['^','v','<','>']
normalized_color = 'g'
plot_normalized = True
##############################################################################

# calculate oct sampling interval
k_1 = 1/lambda_1
k_2 = 1/lambda_2
k_range = k_2-k_1
oct_m_per_pixel = -1/k_range
oct_um_per_pixel = oct_m_per_pixel*1e6

for fom in figures_of_merit:
    normalized_fom = '%s_normalized'%fom
    dat = adf[fom]/adf[normalization_column]/oct_um_per_pixel
    adf[normalized_fom] = dat
    old_label = label_dict[fom]
    new_label = old_label.replace('\mu m','\mathrm{OS}')
    label_dict[normalized_fom] = new_label
    old_ylim = ylim_dict[fom]
    new_ylim = [yl/np.min(adf[normalization_column])/oct_um_per_pixel for yl in old_ylim]
    ylim_dict[normalized_fom] = new_ylim
    
eccentricities_to_omit = []
opf.setup_plots(style=style,font_size=font_size,font=font)

efigs = {}
bfigs = {}
eaxes = {}
baxes = {}

for fig_idx in range(1,len(figures_of_merit)+1):
    fom = figures_of_merit[fig_idx-1]
    f = plt.figure(figsize=figure_size)
    efigs[fom] = f
    lax = f.add_axes(ax_box)
    lax.set_ylabel(label_dict[fom])
    rax = f.add_axes(ax_box)
    nfom = '%s_normalized'%fom
    rax.set_ylabel(label_dict[nfom])
    rax.yaxis.set_label_position('right')
    rax.yaxis.tick_right()
    eaxes[fom] = lax
    eaxes[nfom] = rax
    
for fig_idx in range(1,len(figures_of_merit)+1):
    fom = figures_of_merit[fig_idx-1]
    f = plt.figure(figsize=figure_size)
    bfigs[fom] = f
    lax = f.add_axes(ax_box)
    lax.set_ylabel(label_dict[fom])
    baxes[fom] = lax

    
df_list = []
bad_df_list = []

ecols = edf.columns
acols = adf.columns

for aidx,arow in adf.iterrows():
    adate = arow['date']
    atime = arow['time']

    # in case anything was appended to the original time, just take
    # the first three items delineated by underscores:
    atime = '_'.join(atime.split('_')[:3])

    arow['time'] = atime
    
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

    if len(d.keys())<16:
        bad_df_list.append(d)
    else:
        df_list.append(d)
    
df_all = pd.DataFrame(df_list)
df_all.to_csv('merged_parameters_log.csv')
bad_df_all = pd.DataFrame(bad_df_list)
bad_df_all.to_csv('merged_parameters_log_bad_data.csv')

subject_array = list(df_all['subject'].unique())

temp = list(Line2D.markers.keys())
for item in temp:
    if not item in subject_marker_array:
        subject_marker_array.append(item)


eccentricity_array = sorted(list(df_all['eccentricity'].unique()))
bleaching_array = sorted(list(df_all['bleaching'].unique()))

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
            lax = eaxes[fom]
            lax.plot(eccentricity+x_offset,np.mean(y_arr),color_marker,label=label)
            lax.errorbar(eccentricity+x_offset,np.mean(y_arr),yerr=err(y_arr),ecolor='k',capsize=4)
            lax.set_xlabel('eccentricity (deg)')
            
            if plot_normalized:
                nfom = '%s_normalized'%fom
                y_arr = np.array(ecc_df[nfom])
                ncolor_marker = normalized_color+subject_marker
                rax = eaxes[nfom]
                rax.plot(eccentricity+x_offset,np.mean(y_arr),ncolor_marker,label=None)
                rax.errorbar(eccentricity+x_offset,np.mean(y_arr),yerr=err(y_arr),ecolor='k',capsize=4)

plt.show()
sys.exit()
# done with df, so delete and reuse for bleaching dependence
logx = bleaching_plotting_function==plt.semilogx
    
df = df_all[df_all['subject']==bleaching_subject]
mult = 1
if len(df)>0:
    mult = 2
    for idx,eccentricity in enumerate(eccentricity_array):
        ecc_df = df[df['eccentricity']==eccentricity]
        centered_idx = idx-len(eccentricity_array)//2
        x_offset = centered_idx*b_offset_factor
        
        if eccentricity in eccentricities_to_omit:
            continue

        for bidx,bleaching in enumerate(bleaching_array):
            
            if logx:
                x_offset = centered_idx*bleaching*3.5*x_offset_fraction
                print(x_offset)
                
            if bidx==0:
                label = 'ecc %d'%eccentricity
            else:
                label = None


            bleach_df = ecc_df[ecc_df['bleaching']==bleaching]
            ecc_marker = eccentricity_marker_array[idx]

            for fidx,fom in enumerate(figures_of_merit):
                y_arr = np.array(bleach_df[fom])
                fig = plt.figure(len(figures_of_merit)+fidx+1)
                ax = fig.add_axes(ax_box)
                xbleaching = bleaching
                ax.plot(xbleaching+x_offset,np.mean(y_arr),'k'+ecc_marker,label=label)
                ax.set_xscale('symlog', linthresh=1e0)
                ax.errorbar(xbleaching+x_offset,np.mean(y_arr),yerr=err(y_arr),ecolor='k',capsize=4)
                ax.set_xlabel('bleaching %')
                ax.set_ylabel(label_dict[fom])


for f in range(1,mult*len(figures_of_merit)+1):
    fom = figures_of_merit[(f-1)%len(figures_of_merit)]
    try:
        fig = plt.figure(f)
        leg = plt.legend()
        format_legend(leg)
        
        ax = plt.gca()
        ax.tick_params(direction='in')
        ax.tick_params(left=True)
        ax.tick_params(right=True)
        ax.tick_params(top=True)
        ax.tick_params(bottom=True)
        if not autoscale:
            ax.set_ylim(ylim_dict[fom])
        else:
            ylim = [x for x in ax.get_ylim()]
            if abs(ylim[0])<abs(ylim[1]):
                ylim[0] = ylim[0]
                ylim[1] = ylim[1]*1.1
            elif abs(ylim[0])>abs(ylim[1]):
                ylim[1] = ylim[1]
                ylim[0] = ylim[0]*1.1
            ax.set_ylim(ylim)

        for spine in ['top','bottom','left','right']:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(spine_linewidth)

        fig.tight_layout()

        
        if f<=len(figures_of_merit):
            if ecc_xticks_actual:
                plt.xticks(eccentricity_array)
            plt.savefig('ecc_dependence_%s.svg'%fom)
        else:
            if bleaching_xticks_actual:
                plt.xticks(bleaching_array)
                xlim = (np.min(bleaching_array)-1,np.max(bleaching_array)*2.0)
                plt.xlim(xlim)
            plt.savefig('bleaching_dependence_%s.svg'%fom)
    except Exception as e:
        print(e)
        sys.exit(e)
    
        
plt.show()
