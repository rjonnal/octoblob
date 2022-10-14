from matplotlib import pyplot as plt
import os,sys,glob
import logging
from octoblob import logger
from octoblob import plotting_functions as opf

class Diagnostics:

    def __init__(self,filename):
        self.filename = filename
        self.folder = filename.replace('.unp','')+'_diagnostics'
        os.makedirs(self.folder,exist_ok=True)
        opf.setup_plots()
        
    def save(self,figure_handle,label,index=None):
        if index is None:
            index_string = ''
        else:
            assert type(index)==int
            index_string = '%05d'%index
        outfn = 'diagnostics_%s_%s.png'%(label,index_string)
        outfn = os.path.join(self.folder,outfn)
        figure_handle.savefig(outfn,dpi=opf.print_dpi)
        figure_handle.show()
        plt.show()
