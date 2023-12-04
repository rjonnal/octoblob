import numpy as np
from matplotlib import pyplot as plt
import pathlib,os,glob,sys
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button

png_root = sys.argv[1]
dat_root = sys.argv[2]

png_p = pathlib.Path(png_root)
all_folders = list(png_p.glob('**'))

folders_to_process = []
for f in all_folders:
    print('Checking %s for PNG files.'%f)
    if len(glob.glob(os.path.join(f,'*.png')))>100:
        folders_to_process.append(f)


def viewer(pngf,datf):
    dat_path = pathlib.Path(os.path.join(datf,'BAD_DATA'))
    if os.path.exists(dat_path):
        print('%s already marked BAD_DATA, skipping.'%datf)
        return
    pngs = sorted(glob.glob(os.path.join(pngf,'*.png')))
    fig = plt.figure()
    img_ax = fig.add_axes([0,.5,1.0,.5])
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    
    slider_ax = fig.add_axes([0.1,.4,0.8,0.1])
    slider_ax.set_xticks([])
    slider_ax.set_yticks([])
    buttons_ax = fig.add_axes([0.1,.3,0.8,0.1])
    buttons_ax.set_xticks([])
    buttons_ax.set_yticks([])
    message_ax = fig.add_axes([0,0.1,1.0,0.2])
    message_ax.set_xticks([])
    message_ax.set_yticks([])
    
    npngs = len(pngs)
    def show(idx):
        img = mpimg.imread(pngs[idx])
        imgplot = img_ax.imshow(img)
    
    slider_idx = Slider(ax=slider_ax,label='index',valmin=0,valmax=npngs-1,valstep=1,valinit=10)
    slider_idx.on_changed(show)
        
    def mark_bad(event):
        dat_path.touch()
        plt.close()
        
    btn_bad = Button(buttons_ax,'Mark BAD_DATA',hovercolor='0.95')
    btn_bad.on_clicked(mark_bad)

    message_ax.text(0.0,0.95,'Viewing pngs from %s.'%pngf,ha='left',va='top')
    message_ax.text(0,0.5,'Marking folder %s.'%datf,ha='left',va='top')

    show(10)
    plt.show()

        
for f in folders_to_process:
    f = str(f)
    fpath = pathlib.Path(f)
    toks = fpath.parts
    toks = [t for t in toks]
    bscans_idx = -1
    for idx,tok in enumerate(toks):
        if tok[-7:]=='_bscans':
            bscans_idx = idx
            break
    toks = toks[:bscans_idx+1]
    toks[0] = dat_root
    folder_to_label = '/'.join(toks)
    viewer(f,folder_to_label)
