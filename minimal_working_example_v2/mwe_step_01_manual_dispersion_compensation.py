import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import functions as blobf
from config import dB_clims,dc3_abs_max,dc2_abs_max,noise_roi,dispersion_frame_index,oversample
import pathlib

try:
    filt = sys.argv[1]
except:
    print('Please supply a file or folder name at the command line, i.e., python mweXXX.py XX_YY_ZZ.unp')
    sys.exit()


files = list(pathlib.Path(filt).rglob('*.unp'))
files = [str(f) for f in files]
if len(files)==0:
    files = [filt]

# Define initial parameters
init_dc3 = 0.0
init_dc2 = 0.0

for fn in files:

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
            bscan = blobf.spectra_to_bscan(spectra,[0.0,0.0,dc3,dc2],oversample)
            bscan = np.abs(bscan)
            sharpness = blobf.sharpness(bscan)
            dB = 20*np.log10(np.abs(bscan))
            bscan_dict[(dc3,dc2)] = dB,sharpness
            out = dB
        print('1/sharpness=%0.1f'%(1/sharpness))
        return out,sharpness

    spectra = blobf.get_frame(fn,dispersion_frame_index)
    tag = fn.replace('.unp','')

    try:
        coefs_fn = '%s_mapping_dispersion_coefficients.txt'%tag
        coefs = np.loadtxt(coefs_fn)
        dc3 = coefs[2]
        dc2 = coefs[3]
        coefs_loaded = True
    except FileNotFoundError:
        dc3 = init_dc3
        dc2 = init_dc2
        coefs_loaded = False

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    dB,sharpness = make_bscan(spectra,dc3,dc2)
    if not coefs_loaded:
        ax.set_title('1/sharpness=%0.1f'%(1/sharpness))
    else:
        ax.set_title('1/sharpness=%0.1f\n%s'%(1/sharpness,coefs_fn))
        
    new_clim = np.percentile(dB,(50,99))
    
    img = ax.imshow(dB,clim=new_clim,cmap='gray',aspect='auto')
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.4)

    # Make a horizontal slider to control the dc2
    axdc3 = fig.add_axes([0.25, 0.2, 0.5, 0.03])
    dc3_slider = Slider(
        ax=axdc3,
        label='3rd order disp',
        valmin=-dc3_abs_max,
        valmax=dc3_abs_max,
        valinit=dc3,
    )

    # Make a second horizontally oriented slider to control the dc3
    axdc2 = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    dc2_slider = Slider(
        ax=axdc2,
        label="2nd order disp",
        valmin=-dc2_abs_max,
        valmax=dc2_abs_max,
        valinit=dc2
    )


    # The function to be called anytime a slider's value changes
    def update(val):
        dB,sharpness = make_bscan(spectra, dc3_slider.val, dc2_slider.val)
        img.set_data(dB)
        if not coefs_loaded:
            ax.set_title('1/sharpness=%0.1f'%(1/sharpness))
        else:
            ax.set_title('1/sharpness=%0.1f\n%s'%(1/sharpness,coefs_fn))
        fig.canvas.draw_idle()

    def auto_clim(event):
        dB,sharpness = make_bscan(spectra, dc3_slider.val, dc2_slider.val)
        new_clim = np.percentile(dB,(50,99))
        img.set_clim(new_clim)
        fig.canvas.draw_idle()

    def default_clim(event):
        img.set_clim(dB_clims)
        fig.canvas.draw_idle()


    # register the update function with each slider
    dc3_slider.on_changed(update)
    dc2_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    resetbutton = Button(resetax, 'Reset', hovercolor='0.975')

    # Create a `matplotlib.widgets.Button` to save slider values.
    saveax = fig.add_axes([0.6, 0.025, 0.15, 0.04])
    if coefs_loaded:
        savebutton = Button(saveax, 'Overwrite', hovercolor='0.975')
    else:
        savebutton = Button(saveax, 'Save', hovercolor='0.975')
        
    # Create a `matplotlib.widgets.Button` to set clim automatically.
    autoclimax = fig.add_axes([0.25, 0.025, 0.15, 0.04])
    autoclimbutton = Button(autoclimax, 'Auto clim', hovercolor='0.975')

    # Create a `matplotlib.widgets.Button` to set clim default.
    defaultclimax = fig.add_axes([0.05, 0.025, 0.15, 0.04])
    defaultclimbutton = Button(defaultclimax, 'Default clim', hovercolor='0.975')

    autoclimbutton.on_clicked(auto_clim)
    defaultclimbutton.on_clicked(default_clim)



    def reset(event):
        dc3_slider.reset()
        dc2_slider.reset()

    resetbutton.on_clicked(reset)

    def save(event):
        coefs = [0.0,0.0,dc3_slider.val,dc2_slider.val]
        print('Saving %s to %s_mapping_dispersion_coefficients.txt'%(coefs,tag))
        #np.savetxt('mapping_dispersion_coefficients.txt',coefs)
        np.savetxt('%s_mapping_dispersion_coefficients.txt'%tag,coefs)
        plt.close()
        
    savebutton.on_clicked(save)

    plt.show()

    init_dc3 = dc3_slider.val
    init_dc2 = dc2_slider.val
