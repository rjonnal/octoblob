import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from octoblob.registration_tools import rigid_register
try:
    from fig2gif import GIF
    make_gif = True
    gif_fps = 10
    gif_dpi = 100
except ImportError:
    make_gif = False


make_gif = False

if len(sys.argv)<2:
    print('Usage: python test_project_enface.py input_directory (z1) (z2) (black pct) (white pct) (auto)')
    print('       auto can be 1 or 0; if 1, the program interprets z2-z1 as a slab thickness')
    print('       and automates the generation of angiograms, while setting contrast limits globally')
    print('       using (black pct) and (white pct).')
    sys.exit()

input_directory = sys.argv[1]
output_directory = os.path.join(input_directory,'en_face_projections')
info_directory = os.path.join(output_directory,'info')

z1 = None
z2 = None

if len(sys.argv)>=4:
    z1 = int(sys.argv[2])
    z2 = int(sys.argv[3])

pct_low = 30.0
pct_high = 99.5

if len(sys.argv)>=6:
    pct_low = float(sys.argv[4])
    pct_high = float(sys.argv[5])


auto_mode = False
if len(sys.argv)>=7:
    auto_mode = bool(int(sys.argv[6]))

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

global_clim = np.percentile(vol,(pct_low,pct_high))
profile = vol.mean(2).mean(0)
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
sy_px,sx_px = efp.shape
screen_dpi = 100.0
print_dpi = 300.0
sy_in,sx_in = sy_px/screen_dpi,sx_px/screen_dpi

plt.figure()

plt.figure(figsize=(sx_in,sy_in))
plt.axes([0,0,1,1])

if make_gif:
    gif = GIF(os.path.join(output_directory,'flythrough.gif'),fps=gif_fps,dpi=gif_dpi)
    

for z1 in slab_starts:
    z2 = z1 + dz
    tag = 'en_face_projection_%04d_%04d'%(z1,z2)

    plt.figure(1)
    plt.cla()
    plt.plot(profile)
    plt.axhline(thresh)
    plt.axvline(retina_start)
    plt.axvline(retina_end)
    plt.axvspan(z1,z2,alpha=0.25)
    plt.savefig(os.path.join(info_directory,'profile_%04d_%04d.png'%(z1,z2)))

    plt.figure(2)
    plt.cla()
    

    efp = vol[:,z1:z2,:].mean(1)

    if auto_mode:
        clim = global_clim
    else:
        clim = np.percentile(efp,(pct_low,pct_high))
    plt.imshow(efp,cmap='gray',clim=clim)

    plt.xticks([])
    plt.yticks([])

    plt.savefig(os.path.join(output_directory,'%s.png'%tag),dpi=print_dpi)
    np.save(os.path.join(output_directory,'%s.npy'%tag),efp)
    print('Saving PNG and NPY files to %s.'%output_directory)

    if make_gif:
        gif.add(plt.gcf())
    
    plt.pause(.001)
    
plt.show()

if make_gif:
    gif.make()
