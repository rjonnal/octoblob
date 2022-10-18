from matplotlib import pyplot as plt
import os,sys,glob,time
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

        global npy_dB
        global index,N
        global projection_axis

        projection_axis=0
        
        npy_dB = 1
        
        index = 0
        N = len(files)

        
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

        def last_modified(f):
            epoch_time = os.path.getmtime(f)
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_time))

        def temp_filename(f):
            return '_'.join(tree(f)).strip()
        
        def on_press(event):
            global index,npy_dB,projection_axis
            print('press', event.key)
            sys.stdout.flush()
            if event.key in ['pageup','up','left']:
                index = (index-1)%N
                draw()
            elif event.key in ['pagedown','down','right']:
                index = (index+1)%N
                draw()
            if event.key in ['ctrl+pageup','ctrl+up','ctrl+left']:
                index = (index-10)%N
                draw()
            elif event.key in ['ctrl+pagedown','ctrl+down','ctrl+right']:
                index = (index+10)%N
                draw()
            if event.key in ['ctrl+shift+pageup','ctrl+shift+up','ctrl+shift+left']:
                index = (index-100)%N
                draw()
            elif event.key in ['ctrl+shift+pagedown','ctrl+shift+down','ctrl+shift+right']:
                index = (index+100)%N
                draw()
            elif event.key == 'escape':
                plt.close('all')
            elif event.key == 'z':
                save()
            elif event.key == 'd':
                npy_dB = 1 - npy_dB
                draw()
            elif event.key == 'a':
                projection_axis = (projection_axis+1)%3
                draw()
                
        def draw():
            global index,N
            f = files[index]
            cstr = '(%d/%d):'%(index,N)
            tstr = '(%s)'%last_modified(f)
            
            fig.suptitle(cstr+title(f)+tstr,fontsize=8,ha='left',x=0.0,y=1.0,va='top')
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
            global npy_dB,projection_axis
            dat = np.load(npyfile)
            dat = np.abs(dat)
            if len(dat.shape)==3:
                dat = np.mean(dat,projection_axis)
            
            if npy_dB:
                dat = 20*np.log10(dat)
                clim = (40,90)
            else:
                clim = np.percentile(dat,(5,99.5))
                
            ax.clear()
            ax.imshow(dat,aspect='auto',cmap='gray',clim=clim)
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

        
