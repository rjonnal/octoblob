import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from octoblob.registration_tools import rigid_register
from octoblob.plotting_functions import despine,setup_plots,get_color_cycle

setup_plots(mode='presentation')
cc = get_color_cycle()

try:
    from fig2gif import GIF
    make_gif = True
    gif_fps = 10
    gif_dpi = 100
except ImportError:
    make_gif = False

if len(sys.argv)<2:
    print('Usage: python test_project_enface.py input_directory (z1) (z2) (black dB) (white dB) (auto) (include_profile_plot) (dpi)')
    print('       auto can be 1 or 0; if 1, the program interprets z2-z1 as a slab thickness')
    print('       and automates the generation of angiograms, while setting contrast limits globally')
    print('       using (black pct) and (white pct).')
    print('       include_profile_plot (0 or 1) determines whether the output images contain profile plots; default 0.')
    print('       dpi determines the output dpi; default 100.')
    sys.exit()

input_directory = sys.argv[1]
output_directory = os.path.join(input_directory,'en_face_projections')
info_directory = os.path.join(output_directory,'info')

z1 = None
z2 = None


if len(sys.argv)>=4:
    z1 = int(sys.argv[2])
    z2 = int(sys.argv[3])

dB_low = 40
dB_high =85

if len(sys.argv)>=6:
    dB_low = float(sys.argv[4])
    dB_high = float(sys.argv[5])


auto_mode = False
if len(sys.argv)>=7:
    auto_mode = bool(int(sys.argv[6]))

include_profile_plot = False
if len(sys.argv)>=8:
    include_profile_plot = bool(int(sys.argv[7]))


print_dpi = 100
if len(sys.argv)>=9:
    print_dpi = float(sys.argv[8])

invert = False
if len(sys.argv)>=10:
    invert = bool(int(sys.argv[9]))
    
make_gif = make_gif and auto_mode
    
try:
    os.makedirs(output_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(output_directory)
    except Exception as e:
        print('%s exists; using existing directory')

try:
    os.makedirs(info_directory,exist_ok=True)
except Exception as e:
    try:
        os.mkdir(info_directory)
    except Exception as e:
        print('%s exists; using existing directory')
        
assert os.path.exists(input_directory)

flist = glob.glob(os.path.join(input_directory,'*.npy'))
flist.sort()

vol = []
for f in flist:
    arr = np.load(f)

    # in case data are complex:
    arr = np.abs(arr)
    
    # in case we're working with a stack:
    try:
        assert len(arr.shape)==2
    except AssertionError as ae:
        print('Averaging stack in slow/BM direction.')
        arr = arr.mean(2)
        
    vol.append(arr)

vol = np.array(vol)

global_clim = (dB_low,dB_high)
profile = vol.mean(2).mean(0)
profile[:5] = profile.min()
profile[-5:] = profile.min()

thresh = profile.min()+profile.std()*0.1
retina_start = np.where(profile>thresh)[0][0]
retina_end = np.where(profile>thresh)[0][-1]


if z1 is None or z2 is None:
    plt.subplot(1,3,1)
    plt.imshow(vol.mean(0))
    plt.subplot(1,3,2)
    plt.imshow(vol.mean(2).T)
    plt.subplot(1,3,3)
    plt.plot(profile,range(len(profile)))
    plt.ylim((len(profile),0))
    plt.suptitle('Please enter axial extent in console window.')
    plt.pause(.1)

    z1 = int(input('Enter z1: '))
    z2 = int(input('Enter z2: '))

dz = z2-z1
if auto_mode:
    slab_starts = range(retina_start,retina_end)
else:
    slab_starts = [z1]
    
efp = vol[:,z1:z2,:].mean(1)

efp = 20*np.log10(efp)

sy_px,sx_px = efp.shape
screen_dpi = 100.0
sy_in,sx_in = sy_px/screen_dpi,sx_px/screen_dpi


if include_profile_plot:
    fig_both = plt.figure(figsize=(6,4),dpi=screen_dpi)
    ax_plot = plt.axes([.13,.13,.35,.85])
    ax_image = plt.axes([.5,0,.5,1])
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    
else:
    fig_plot = plt.figure(figsize=(4,4))
    ax_plot = plt.axes([.2,.13,.8,.85])
    fig_image = plt.figure(figsize=(sx_in,sy_in))
    ax_image = fig_image.add_axes([0,0,1,1])
    

if make_gif:
    if invert:
        giffn = 'flythrough_inverted.gif'
    else:
        giffn = 'flythrough.gif'
        
    gif = GIF(os.path.join(output_directory,giffn),fps=gif_fps,dpi=gif_dpi)
    

for z1 in slab_starts:
    z2 = z1 + dz

    if invert:
        tag = 'en_face_projection_inverted_%04d_%04d'%(z1,z2)
    else:
        tag = 'en_face_projection_%04d_%04d'%(z1,z2)

    ax_plot.clear()
    ax_plot.plot(profile,color=cc[0])
    ax_plot.axhline(thresh,color=cc[2])
    ax_plot.axvline(retina_start,color=cc[1])
    ax_plot.axvline(retina_end,color=cc[1])
    ax_plot.axvspan(z1,z2,alpha=0.25,color=cc[3])
    despine(ax_plot)
    ax_plot.set_xlabel('depth (px)')
    ax_plot.set_ylabel('amplitude (ADU)')



    if not include_profile_plot:
        plt.savefig(os.path.join(info_directory,'profile_%04d_%04d.png'%(z1,z2)))


    efp = vol[:,z1:z2,:].mean(1)
    efp = 20*np.log10(efp)

    #print(efp.max(),efp.min())
    #sys.exit()
    
    if auto_mode:
        clim = global_clim
    else:
        clim = (dB_low,dB_high)

    if invert:
        cmap = 'gray_r'
    else:
        cmap = 'gray'
        
    ax_image.clear()
    imh = ax_image.imshow(efp,cmap=cmap,clim=clim)

    ax_image.set_xticks([])
    ax_image.set_yticks([])

    despine(ax_image)
    
    plt.savefig(os.path.join(output_directory,'%s.png'%tag),dpi=print_dpi)
    np.save(os.path.join(output_directory,'%s.npy'%tag),efp)
    
    print('Saving PNG and NPY files to %s.'%output_directory)

    if make_gif:
        gif.add(plt.gcf())
    
    plt.pause(.001)
    
if not make_gif:
    plt.show()
else:
    plt.close('all')

if make_gif:
    gif.make()
