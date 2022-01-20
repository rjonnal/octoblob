import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys,os
import octoblob.plotting_functions as opf
from matplotlib.lines import Line2D


##############################################################################
bleaching_subject = 1
plot_folder = './summary_plots'

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
zero_overshoot = 1.5
ylim_dict['vmin_0_20'] = [-4,zero_overshoot]
ylim_dict['vmax_20_40'] = [-zero_overshoot,3]
ylim_dict['vmean_20_40'] = [-zero_overshoot,2]
ylim_dict['amin_0_50'] = [-2.5,zero_overshoot]
ylim_dict['amax_0_50'] = [-zero_overshoot,2.5]
ylim_dict['std_0_50'] = [-zero_overshoot,3]
ylim_dict['mad_0_50'] = [-zero_overshoot,3]

legendloc_dict = {}
legendloc_dict['vmin_0_20'] = 4
legendloc_dict['vmax_20_40'] = 1
legendloc_dict['vmean_20_40'] = 1
legendloc_dict['amin_0_50'] = 1
legendloc_dict['amax_0_50'] = 1
legendloc_dict['std_0_50'] = 1
legendloc_dict['mad_0_50'] = 1


# edit filenames below as needed:
edf = pd.read_csv('experimental_parameters_log.csv')
adf = pd.read_csv('analysis_parameters_log.csv')

figure_size = (4,4) # (width_inches, height_inches)
ax_box = [.18,.15,.77,.80]
nax_box = [.18,.15,.64,.80]
ax_box_log_zero = [.18,.15,.15,.80]
ax_box_log_nonzero = [.35,.15,.47,.80]

font = 'Arial'
font_size = 10
screen_dpi = 100
print_dpi = 300
plot_linewidth = 2

# spines (lines at edges of plot)
spine_color = 'k'
spine_linewidth = 1

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
x_offset_fraction = 0.015

# Determines whether to use automatic x-ticks or the values of x in the data:
ecc_xticks_actual = False
bleaching_xticks_actual = True

# plot markers
normalized_color = 'g'
plot_normalized = True
normalize_text = lambda x: x.replace('\mu m','\mathrm{OS}')


subject_marker_array = ['o','s','d']
ecc_marker_array = ['o','s','d','*']

eccentricities_to_omit = []

bleaching_logx = True

rebuild = False
##############################################################################


os.makedirs(plot_folder,exist_ok=True)

# Basic plot setup:
opf.setup_plots(style=style,font_size=font_size,font=font)
plt.rc('axes', axisbelow=True)

# Function for calculating error bars:
def err(vec,mode='SEM'):
    if mode=='SEM':
        return np.std(vec)/np.sqrt(len(vec))
    elif mode=='SD':
        return np.std(vec)

# Function for formatting legends:
def format_legend(leg):
    leg.get_frame().set_edgecolor(legend_edgecolor)
    leg.get_frame().set_facecolor(legend_facecolor)
    leg.get_frame().set_linewidth(legend_linewidth)
    leg.get_frame().set_alpha(legend_alpha)


    
#################################
# calculate oct sampling interval
#################################
k_1 = 1/lambda_1
k_2 = 1/lambda_2
k_range = k_2-k_1
oct_m_per_pixel = -1/k_range
oct_um_per_pixel = oct_m_per_pixel*1e6

##########################
# merge the two DataFrames
##########################
df_list = []
bad_df_list = []

ecols = edf.columns
acols = adf.columns

if rebuild:
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
else:
    df_all = pd.read_csv('merged_parameters_log.csv')


def unique(LL):
    output = []
    for L in LL:
        for item in L:
            if not item in output:
                output.append(item)
    return output

subject_array = sorted(list(df_all['subject'].unique()))
ecc_array = sorted(list(df_all['eccentricity'].unique()))
bleaching_array = sorted(list(df_all['bleaching'].unique()))
max_bleaching = np.max(bleaching_array)

try:
    assert len(subject_array)<=len(subject_marker_array)
except Exception:
    print('Not enough subject markers in subject_marker_array.')
try:
    assert len(ecc_array)<=len(ecc_marker_array)
except Exception:
    print('Not enough ecc markers in ecc_marker_array.')


ecc_ds_dict = {}
b_ds_dict = {}

exr = np.max(ecc_array)-np.min(ecc_array)
bxr = np.max(bleaching_array)-np.min(bleaching_array)

# Build the ecc data plots:
for fom in figures_of_merit:
    fig = plt.figure(figsize=figure_size)
    
    if plot_normalized:
        ax = fig.add_axes(nax_box)
        nax = ax.twinx()
        nax.yaxis.set_label_position('right')
        nax.yaxis.tick_right()

        nax.yaxis.label.set_color(normalized_color)
        nax.tick_params(axis='y', colors=normalized_color)
        #nax.spines['right'].set_color(normalized_color)

    else:
        ax = fig.add_axes(ax_box)
        
    # each subject is one series in this plot
    for sidx,subject in enumerate(subject_array):
        xo = exr*x_offset_fraction*(sidx-len(subject_array)//2)
        marker = subject_marker_array[sidx]
        color_marker = 'k'+marker
        ncolor_marker = normalized_color+subject_marker_array[sidx]
        ymean = []
        yerr = []
        nymean = []
        nyerr = []
        for ecc in ecc_array:
            rows = df_all[(df_all['bleaching']==66)&(df_all['subject']==subject)&(df_all['eccentricity']==ecc)]
            ydat = rows[fom]
            ymean.append(np.mean(ydat))
            yerr.append(err(ydat))
            norm = rows[normalization_column]
            nydat = ydat/norm
            nymean.append(np.mean(nydat))
            nyerr.append(err(nydat))
            
            
        ax.plot(ecc_array+xo,ymean,marker=marker,markerfacecolor='k',
                label='subject %d'%subject,linestyle='none',markeredgecolor='none')
        ax.errorbar(ecc_array+xo,ymean,yerr=yerr,ecolor='k',capsize=4,linestyle='none')

        if plot_normalized:
            nax.plot(ecc_array+xo,nymean,marker=marker,markerfacecolor=normalized_color,
                     linestyle='none',markeredgecolor='none')
            #nax.plot(ecc_array+xo,nymean,ncolor_marker)
            nax.errorbar(ecc_array+xo,nymean,yerr=nyerr,ecolor=normalized_color,capsize=4,linestyle='none')

    ax.set_ylabel(label_dict[fom])
    ax.set_ylim(ylim_dict[fom])
    if plot_normalized:
        nax.set_ylabel(normalize_text(label_dict[fom]))
        nax.set_yticks(ax.get_yticks())
        #nax.set_yticks([])
        nax.set_ylim(ylim_dict[fom])
        
    for spine in ['top','bottom','left','right']:
        nax.spines[spine].set_visible(False)
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(spine_linewidth)

    ax.set_axisbelow(True)
    if plot_normalized:
        nax.set_axisbelow(True)
    leg = ax.legend(loc=legendloc_dict[fom])
    format_legend(leg)
    nax.yaxis.grid(False)
    nax.xaxis.grid(False)
    ax.set_xlabel('eccentricity ($^\circ$)')
    plt.savefig(os.path.join(plot_folder,'ecc_dependence_%s.svg'%fom))


    
# Build the bleaching data plots:
ba_xticks = [b for b in bleaching_array if b>0]
for fom in figures_of_merit:
    fig = plt.figure(figsize=figure_size)

    if bleaching_logx:
        lax = fig.add_axes(ax_box_log_zero)
        lax.set_xlim([-.5,.5])
        lax.set_xticks([0])
        rax = fig.add_axes(ax_box_log_nonzero)
        rax.yaxis.set_label_position('right')
        rax.yaxis.tick_right()
    else:
        lax = fig.add_axes(ax_box)
    
    # each ecc is one series in this plot
    for eidx,ecc in enumerate(ecc_array):
        centered_idx = eidx-len(ecc_array)//2+0.5
        xo = bxr*x_offset_fraction*centered_idx*0.1
        marker = ecc_marker_array[eidx]
        ymean = []
        yerr = []
        for bleaching in bleaching_array:
            rows = df_all[(df_all['bleaching']==bleaching)&(df_all['subject']==bleaching_subject)&(df_all['eccentricity']==ecc)]
            ydat = rows[fom]
            ymean.append(np.mean(ydat))
            yerr.append(err(ydat))

        if bleaching_logx:
            for b,ym,ye in zip(bleaching_array,ymean,yerr):
                xolog = b*x_offset_fraction*centered_idx*5
                if b==np.max(bleaching_array):
                    label = '$%d^\circ$'%ecc
                else:
                    label = None
                if b==0:
                    lax.plot(b+xo,ym,marker=marker,markerfacecolor='w',
                             linestyle='none',markeredgecolor='k')
                    lax.errorbar(b+xo,ym,yerr=ye,ecolor='k',capsize=4,linestyle='none')
                else:
                    rax.semilogx(b+xolog,ym,marker=marker,markerfacecolor='w',
                                 label=label,linestyle='none',markeredgecolor='k')
                    rax.errorbar(b+xolog,ym,yerr=ye,ecolor='k',capsize=4,linestyle='none')
                    
        else:
            lax.plot(bleaching_array+xo,ymean,marker=marker,markerfacecolor='k',
                     linestyle='none',markeredgecolor='none',label='$%d^\circ$'%ecc)

            

    lax.set_ylabel(label_dict[fom])
    lax.set_ylim(ylim_dict[fom])
    if bleaching_logx:
        rax.set_ylim(ylim_dict[fom])
        rax.set_xticks(ba_xticks)
        rax.set_xticklabels(ba_xticks)
        
    for spine in ['top','bottom','left','right']:
        lax.spines[spine].set_color(spine_color)
        lax.spines[spine].set_linewidth(spine_linewidth)
        if bleaching_logx:
            rax.spines[spine].set_color(spine_color)
            rax.spines[spine].set_linewidth(spine_linewidth)
            if spine=='left':
                rax.spines[spine].set_visible(False)
            if spine=='right':
                lax.spines[spine].set_visible(False)

    if bleaching_logx:
        leg = rax.legend(loc=legendloc_dict[fom])
        rax.set_xlabel('bleaching (%)                   ')
    else:
        leg = lax.legend(loc=legendloc_dict[fom])
        lax.set_xlabel('bleaching (%)')
        
    format_legend(leg)
    plt.savefig(os.path.join(plot_folder,'bleaching_dependence_%s.svg'%fom))


    
plt.show()
