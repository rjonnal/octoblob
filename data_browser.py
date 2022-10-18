from matplotlib import pyplot as plt
import os,sys,glob
import logging
from octoblob import logger
import pathlib
import numpy as np
import matplotlib.image as mpimg

class Browser:

    def browse(self,root='.',file_filters=['*.npy','*.png'],figsize=(6,6)):

        save_folder = 'browser_saves'
        os.makedirs(save_folder,exist_ok=True)
        
        files = []
        for ff in file_filters:
            files = files + list(pathlib.Path(root).rglob(ff))

        files = list(files)
        files = sorted(files)

        def tree(f):
            head = str(f)
            items = []
            while len(head):
                head,tail = os.path.split(head)
                items.append(tail)
            items = items[::-1]
            return items

        def title(f):
            t = tree(f)
            out = ''
            for item_index,item in enumerate(t):
                out = out + item_index*'  ' + item + '\n'
            return out

        def temp_filename(f):
            return '_'.join(tree(f)).strip()
        
        N = len(files)

        global index
        index = 0
        
        def on_press(event):
            global index
            print('press', event.key)
            sys.stdout.flush()
            if event.key in ['pageup','up','left']:
                index = (index+1)%N
                draw()
            elif event.key in ['pagedown','down','right']:
                index = (index-1)%N
                draw()
            elif event.key == 'escape':
                plt.close('all')
            elif event.key == 'z':
                save()

        def draw():
            f = files[index]
            fig.suptitle(title(f),fontsize=8,ha='left',x=0.0,y=1.0,va='top')
            ext = os.path.splitext(f)[1]
            if ext.lower()=='.npy':
                print('npy')
                npydraw(f)
            elif ext.lower()=='.png':
                print('png')
                pngdraw(f)
            

        def pngdraw(pngfile):
            img = mpimg.imread(pngfile)
            ax.clear()
            ax.imshow(img,aspect='auto',cmap='gray')
            fig.canvas.draw()

        def npydraw(npyfile):
            dat = np.load(npyfile)
            dat = np.abs(dat)
            ax.clear()
            ax.imshow(dat,aspect='auto',cmap='gray')
            fig.canvas.draw()

        def save():
            f = files[index]
            ffn = os.path.join(save_folder,temp_filename(f))
            print('Saving current image to %s.'%ffn)
            plt.savefig(ffn,dpi=300)
            
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        fig.canvas.mpl_connect('key_press_event', on_press)
        draw()
        plt.show()

        
