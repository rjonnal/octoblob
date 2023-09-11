import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import functions as blobf

dB_clims = (40,90)
dc3_abs_max = 1e-7
dc2_abs_max = 1e-4

noise_roi = [10,100,10,100]

try:
    fn = sys.argv[1]
except:
    print('Please supply the filename at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
    sys.exit()


# cache results for quick recall:
bscan_dict = {}
    
def make_bscan(spectra,dc3,dc2):
    dc3 = float('%0.3g'%dc3)
    dc2 = float('%0.3g'%dc2)
    print('coefs = %s'%[0.0,0.0,dc3,dc2])
    try:
        out,sharpness = bscan_dict[(dc3,dc2)]
        #print('Using cached bscan.')
    except KeyError:
        #print('Calculating bscan.')
        bscan = blobf.spectra_to_bscan(spectra,[0.0,0.0,dc3,dc2])
        bscan = np.abs(bscan)
        sharpness = blobf.sharpness(bscan)
        dB = 20*np.log10(np.abs(bscan))
        bscan_dict[(dc3,dc2)] = dB,sharpness
        out = dB
    print('1/sharpness=%0.1f'%(1/sharpness))
    return out,sharpness

spectra = blobf.get_frame(fn,50)

# Define initial parameters
init_dc3 = 0.0
init_dc2 = 0.0

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
dB,sharpness = make_bscan(spectra,init_dc3,init_dc2)
ax.set_title('1/sharpness=%0.1f'%(1/sharpness))
img = ax.imshow(dB,clim=dB_clims,cmap='gray',aspect='auto')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.4)

# Make a horizontal slider to control the dc2
axdc3 = fig.add_axes([0.25, 0.2, 0.5, 0.03])
dc3_slider = Slider(
    ax=axdc3,
    label='3rd order disp',
    valmin=-dc3_abs_max,
    valmax=dc3_abs_max,
    valinit=init_dc3,
)

# Make a second horizontally oriented slider to control the dc3
axdc2 = fig.add_axes([0.25, 0.1, 0.5, 0.03])
dc2_slider = Slider(
    ax=axdc2,
    label="2nd order disp",
    valmin=-dc2_abs_max,
    valmax=dc2_abs_max,
    valinit=init_dc2
)


# The function to be called anytime a slider's value changes
def update(val):
    dB,sharpness = make_bscan(spectra, dc3_slider.val, dc2_slider.val)
    img.set_data(dB)
    ax.set_title('1/sharpness=%0.1f'%(1/sharpness))
    fig.canvas.draw_idle()


# register the update function with each slider
dc3_slider.on_changed(update)
dc2_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
resetbutton = Button(resetax, 'Reset', hovercolor='0.975')

# Create a `matplotlib.widgets.Button` to save slider values.
saveax = fig.add_axes([0.65, 0.025, 0.1, 0.04])
savebutton = Button(saveax, 'Save', hovercolor='0.975')



def reset(event):
    dc3_slider.reset()
    dc2_slider.reset()

resetbutton.on_clicked(reset)

def save(event):
    coefs = [0.0,0.0,dc3_slider.val,dc2_slider.val]
    print('Saving %s to mapping_dispersion_coefficients.txt'%coefs)
    np.savetxt('mapping_dispersion_coefficients.txt',coefs)

savebutton.on_clicked(save)

plt.show()

