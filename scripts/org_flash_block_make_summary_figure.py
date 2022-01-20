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
ecc_ylim_dict = {}
ecc_ylim_dict['vmin_0_20'] = [-4,0.0]
ecc_ylim_dict['vmax_20_40'] = [-0.0,3]
ecc_ylim_dict['vmean_20_40'] = [-0.5,2.0]
ecc_ylim_dict['amin_0_50'] = [-2.25,0.0]
ecc_ylim_dict['amax_0_50'] = [-0.0,2.5]
ecc_ylim_dict['std_0_50'] = [-0.0,3]
ecc_ylim_dict['mad_0_50'] = [-0.0,3]

ecc_legendloc_dict = {}
ecc_legendloc_dict['vmin_0_20'] = 4
ecc_legendloc_dict['vmax_20_40'] = 0
ecc_legendloc_dict['vmean_20_40'] = 0
ecc_legendloc_dict['amin_0_50'] = 4
ecc_legendloc_dict['amax_0_50'] = 0
ecc_legendloc_dict['std_0_50'] = 0
ecc_legendloc_dict['mad_0_50'] = 0


bleaching_ylim_dict = {}
bleaching_ylim_dict['vmin_0_20'] = [-2.75,0.0]
bleaching_ylim_dict['vmax_20_40'] = [-0.0,3]
bleaching_ylim_dict['vmean_20_40'] = [-0.5,1.75]
bleaching_ylim_dict['amin_0_50'] = [-2.5,0.0]
bleaching_ylim_dict['amax_0_50'] = [-0.0,2.5]
bleaching_ylim_dict['std_0_50'] = [-0.0,3]
bleaching_ylim_dict['mad_0_50'] = [-0.0,3]

bleaching_legendloc_dict = {}
bleaching_legendloc_dict['vmin_0_20'] = 0
bleaching_legendloc_dict['vmax_20_40'] = 0
bleaching_legendloc_dict['vmean_20_40'] = 4
bleaching_legendloc_dict['amin_0_50'] = 4
bleaching_legendloc_dict['amax_0_50'] = 0
bleaching_legendloc_dict['std_0_50'] = 0
bleaching_legendloc_dict['mad_0_50'] = 0


# edit filenames below as needed:
edf = pd.read_csv('experimental_parameters_log.csv')
adf = pd.read_csv('analysis_parameters_log.csv')

figure_size = (4,4) # (width_inches, height_inches)
ax_box = [.18,.15,.77,.80]
nax_box = [.18,.15,.64,.80]
ax_box_log_zero = [.18,.15,.17,.80]
ax_box_log_nonzero = [.37,.15,.58,.80]

font = 'Arial'
font_size = 10
screen_dpi = 100
print_dpi = 300
plot_linewidth = 2

errorbar_capsize = 2

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
x_offset_fraction = 0.015

# Determines whether to use automatic x-ticks or the values of x in the data:
ecc_xticks_actual = False
bleaching_xticks_actual = True

# plot markers
main_marker_color = [0.0,0.0,0.0]
normalized_color = [0.5,0.5,0.5]
face_color = [1.0,1.0,1.0]

plot_normalized = True
normalize_text = lambda x: x.replace('\mu m','\mathrm{OS}')


subject_marker_array = ['o','s','X']
subject_marker_size_array = [7,7,9]
ecc_marker_array = ['o','s','X','P']
ecc_marker_size_array = [7,7,9,9]
eccentricities_to_omit = []

bleaching_logx = True

# Set rebuild to False to speed things up, if you've already run this before;
# When False, the script reads the merged CSV file instead of re-merging them
# Make sure to set this to True if there have been any changes in the analysis
# or experimental parameter logs:
rebuild = True
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


def draw_break(ax_box,side,angle=30,relative_length=0.02):
    if side=='left':
        x0 = ax_box[0]
    elif side=='right':
        x0 = ax_box[2]+ax_box[0]
    y0list = [ax_box[1],ax_box[1]+ax_box[3]]
    ax = plt.gcf().add_axes([0,0,1,1])
    ax.axis('off')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    ax.set_alpha(0.0)
    ax.autoscale(False)
    
    theta = angle*np.pi/180.0
    dx = relative_length*np.sin(theta)
    dy = relative_length*np.cos(theta)
    for y0 in y0list:
        ax.plot([x0-dx,x0+dx],[y0-dy,y0+dy],linestyle='-',linewidth=spine_linewidth,color=main_marker_color)
        
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
            
            
        ax.plot(ecc_array+xo,ymean,marker=marker,markerfacecolor=main_marker_color,
                label='subject %d'%subject,linestyle='none',markeredgecolor='none',markersize=subject_marker_size_array[sidx])
        ax.errorbar(ecc_array+xo,ymean,yerr=yerr,ecolor=main_marker_color,capsize=errorbar_capsize,linestyle='none')

        if plot_normalized:
            nax.plot(ecc_array+xo,nymean,marker=marker,markerfacecolor=normalized_color,
                     linestyle='none',markeredgecolor='none',markersize=subject_marker_size_array[sidx])
            nax.errorbar(ecc_array+xo,nymean,yerr=nyerr,ecolor=normalized_color,capsize=errorbar_capsize,linestyle='none')

    ax.set_ylabel(label_dict[fom])
    ax.set_ylim(ecc_ylim_dict[fom])
    if plot_normalized:
        nax.set_ylabel(normalize_text(label_dict[fom]))
        nax.set_yticks(ax.get_yticks())
        #nax.set_yticks([])
        nax.set_ylim(ecc_ylim_dict[fom])
        
    for spine in ['top','bottom','left','right']:
        nax.spines[spine].set_visible(False)
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(spine_linewidth)

    ax.set_axisbelow(True)
    if plot_normalized:
        nax.set_axisbelow(True)
    leg = ax.legend(loc=ecc_legendloc_dict[fom])
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
        #rax.yaxis.set_label_position('right')
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
                    lax.plot(b+xo,ym,marker=marker,markerfacecolor=face_color,
                             linestyle='none',markeredgecolor=main_marker_color,markersize=ecc_marker_size_array[eidx])
                    lax.errorbar(b+xo,ym,yerr=ye,ecolor=main_marker_color,capsize=errorbar_capsize,linestyle='none')
                else:
                    rax.semilogx(b+xolog,ym,marker=marker,markerfacecolor=face_color,
                                 label=label,linestyle='none',markeredgecolor=main_marker_color,
                                 markersize=ecc_marker_size_array[eidx])
                    rax.errorbar(b+xolog,ym,yerr=ye,ecolor=main_marker_color,capsize=errorbar_capsize,linestyle='none')
                    
        else:
            lax.plot(bleaching_array+xo,ymean,marker=marker,markerfacecolor=main_marker_color,
                     linestyle='none',markeredgecolor='none',label='$%d^\circ$'%ecc,markersize=ecc_marker_size_array[eidx])
            lax.errorbar(bleaching_array+xo,ymean,yerr=yerr,ecolor=main_marker_color,capsize=errorbar_capsize,linestyle='none')

            

    lax.set_ylabel(label_dict[fom])
    lax.set_ylim(bleaching_ylim_dict[fom])
    if bleaching_logx:
        rax.set_ylim(bleaching_ylim_dict[fom])
        rax.set_xticks(ba_xticks)
        rax.set_xticklabels(ba_xticks)
        
    if bleaching_logx:
        draw_break(ax_box_log_nonzero,'left')
        draw_break(ax_box_log_zero,'right')
        
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
        leg = rax.legend(loc=bleaching_legendloc_dict[fom])
        label_x = (ax_box_log_zero[0]+ax_box_log_nonzero[0]+ax_box_log_nonzero[2])/2.0
        fig.text(label_x,ax_box_log_zero[1]-0.075,'bleaching (%)',ha='center',va='top')
        #rax.set_xlabel('bleaching (%)                   ')
        rax.set_yticklabels([])
    else:
        leg = lax.legend(loc=bleaching_legendloc_dict[fom])
        lax.set_xlabel('bleaching (%)')
        
    format_legend(leg)
    
    plt.savefig(os.path.join(plot_folder,'bleaching_dependence_%s.svg'%fom))


    
plt.show()
