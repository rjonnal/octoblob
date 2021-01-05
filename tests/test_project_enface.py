import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob

if len(sys.argv)<2:
    print('Usage: python test_project_enface.py input_directory')
    sys.exit()

input_directory = sys.argv[1]
output_directory = os.path.join(input_directory,'en_face_projections')

os.makedirs(output_directory,exist_ok=True)

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
profile = vol.mean(2).mean(0)
plt.subplot(1,2,1)
plt.imshow(vol.mean(0))
plt.title('Please enter axial extent in console window.')
plt.subplot(1,2,2)
plt.plot(profile,range(len(profile),0,-1))
plt.pause(.1)

z1 = int(input('Enter z1: '))
z2 = int(input('Enter z2: '))

tag = 'en_face_projection_%04d_%04d'%(z1,z2)

efp = vol[:,z1:z2,:].mean(1)

sy_px,sx_px = efp.shape
screen_dpi = 100.0
print_dpi = 300.0
sy_in,sx_in = sy_px/screen_dpi,sx_px/screen_dpi

plt.figure(figsize=(sx_in,sy_in))
plt.axes([0,0,1,1])

plt.imshow(efp,cmap='gray')

plt.xticks([])
plt.yticks([])

plt.savefig(os.path.join(output_directory,'%s.png'%tag),dpi=print_dpi)
np.save(os.path.join(output_directory,'%s.npy'%tag),efp)
print('Saving PNG and NPY files to %s.'%output_directory)

plt.show()

