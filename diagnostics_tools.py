from matplotlib import pyplot as plt
import os,sys,glob
import logging
from octoblob import logger
import inspect

class Diagnostics:

    def __init__(self,filename,limit=3):
        self.filename = filename
        print(filename)
        self.folder = filename.replace('.unp','')+'_diagnostics'
        os.makedirs(self.folder,exist_ok=True)
        self.limit = limit
        self.dpi = 150
        self.figures = {}
        self.labels = {}
        self.counts = {}
        self.done = []
        self.fig = plt.figure()

    def log(title,header,data,fmt,clobber):
        print(title)
        print(header)
        print(fmt%data)
        
    def save(self,figure_handle,label=None,ignore_limit=False):
        label = inspect.currentframe().f_back.f_code.co_name
        if label in self.done:
            return
        
        subfolder = os.path.join(self.folder,label)
        index = self.counts[label]
        
        if index<self.limit or ignore_limit:
            outfn = os.path.join(subfolder,'%s_%05d.png'%(label,index))
            figure_handle.savefig(outfn,dpi=self.dpi)
            self.counts[label]+=1
        else:
            self.done.append(label)
            
        plt.close(figure_handle.number)
            

    def figure(self,figsize=(4,4),dpi=100,label=None):
        label = inspect.currentframe().f_back.f_code.co_name
        subfolder = os.path.join(self.folder,label)
        if not label in self.counts.keys():
            self.counts[label] = 0
            os.makedirs(subfolder,exist_ok=True)
        self.fig.clear()
        self.fig.set_size_inches(figsize[0],figsize[1], forward=True)
        #out = plt.figure(figsize=figsize,dpi=dpi)
        return self.fig
    
