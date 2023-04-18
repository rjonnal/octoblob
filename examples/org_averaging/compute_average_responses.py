from matplotlib import pyplot as plt
import numpy as np
import sys,os,glob,shutil
import logging
import octoblob.functions as blobf
import octoblob.org_tools as blobo
import octoblob.plotting_functions as blobp
import pathlib

root_folder = '.'

# When you run plot_general_org.py or plot_cone_org.py, you save results (using the Enter key)
# to two locations: the working folder (wherever you run the script) and also into the *_bscans/org
# subfolders. In both places, a subfolder called layer_velocities_results is created, and the
# results are stored there. In other words, they are duplicated. (I'm not sure why I left this
# unnecessary duplication, but now that it's there different downstream scripts assume different
# locations, so we have to keep it this way for now.)

# This program can take a list of the *_velocity.npy files and generate an average result from these.
# If the files are not specified (in the velocity_files list below), then the program will find
# all of the *_velocity.npy files below the working folder and average these. It will avoid
# duplicates (such as those in the working folder being duplicates of those in the org subfolders)
# and only average unique responses.


# make velocity_files = [] for automatic detection of all velocity files
#velocity_files = [
#    '16_58_12_bscans/org/layer_velocities_results/16_58_12_bscans_org_16_176_182_188_velocity.npy',
#    '16_53_25_bscans/org/layer_velocities_results/16_53_25_bscans_org_17_174_172_178_velocity.npy']
velocity_files = []

if len(velocity_files)==0:
    velocity_files = [str(v) for v in pathlib.Path(root_folder).rglob('*velocity.npy')]

stimulus_index = 100

# in the average plot, do you want the component plots potted too?
plot_background = True

# figure dimensions/dpi
screen_dpi = 50
panel_width_in = 4.0
panel_height_in = 4.0

# parameters for the response plot lines
main_alpha = 1.0
main_linewidth = 1.5

background_alpha = 0.25
background_linewidth = 1.5

single_color = 'k'
average_color = 'b'

tlim = (-0.04,0.04) # time limits for plotting ORG in s
vlim = (-5,5) # velocity limits for plotting in um/s

z_um_per_pixel = 3.0

single_responses = []
bscans = []
used = []

def match(a,b):
    return a.find(b)>-1 or b.find(a)>-1

for vf in velocity_files:
    short_fn = os.path.split(vf)[1]
    if not any([match(short_fn,fn) for fn in used]):
        single_responses.append(np.load(vf))
        used.append(short_fn)

single_responses = np.array(single_responses)
average_response = np.mean(single_responses,axis=0)

n_files = len(single_responses)
n_plots = n_files+1
n_t = single_responses.shape[1]

t = np.arange(n_t)-stimulus_index
t = t*2.5e-3+10e-3

plt.figure(figsize=(panel_width_in*n_plots,panel_height_in),dpi=screen_dpi)
for row in range(0,n_files):
    ax = plt.subplot(1,n_plots,row+1)
    ax.plot(t,single_responses[row,:],color=single_color,linewidth=main_linewidth,alpha=main_alpha)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$v_{OS}$ ($\mu m$/s)')
    ax.plot([0.0,0.0],[vlim[0]*.75,vlim[1]*.75],color='g',linestyle='--')
    ax.set_xlim(tlim)
    ax.set_ylim(vlim)
    ax.grid(False)
    blobp.despine(ax,'btlr')
    
ax = plt.subplot(1,n_plots,n_files+1)
ax.plot(t,average_response,color=average_color,linewidth=main_linewidth,alpha=main_alpha)
ax.set_xlabel('time (s)')
ax.set_ylabel('$\overline{v_{OS}}$ ($\mu m$/s)')
ax.plot([0.0,0.0],[vlim[0]*.75,vlim[1]*.75],color='g',linestyle='--')
ax.set_xlim(tlim)
ax.set_ylim(vlim)
ax.grid(False)
blobp.despine(ax,'btlr')

if plot_background:
    for row in range(0,n_files):
        ax.plot(t,single_responses[row,:],color=single_color,linewidth=background_linewidth,alpha=background_alpha)

plt.savefig('average_response.png',dpi=300)
plt.savefig('average_response.pdf')
plt.show()
