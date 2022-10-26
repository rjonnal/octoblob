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
        self.counts = {}
        self.limit = limit
        self.dpi = 150
        self.figures = {}

    def log(title,header,data,fmt,clobber):
        print(title)
        print(header)
        print(fmt%data)
        
    def save(self,figure_handle,label=None,ignore_limit=False):

        if label is None:
            try:
                label = inspect.currentframe().f_back.f_code.co_name
            except Exception as e:
                print(e)
                label = 'unknown'
        
        subfolder = os.path.join(self.folder,label)
        if not label in self.counts.keys():
            self.counts[label] = 0
            os.makedirs(subfolder,exist_ok=True)
        if not label in self.figures.keys():
            self.figures[label] = plt.figure()

        plt.figure(self.figures[label].number)
        plt.clf()
        index = self.counts[label]
        
        if index<self.limit or ignore_limit:
            outfn = os.path.join(subfolder,'%s_%05d.png'%(label,index))
            print(outfn)
            figure_handle.savefig(outfn,dpi=self.dpi)
            #figure_handle.close()
            plt.close(figure_handle)
            self.counts[label]+=1

        plt.close(self.figures[label].number)
