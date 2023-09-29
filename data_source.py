from builtins import *
import sys,os,time
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.io as sio
import glob
import shutil
import logging
from octoblob import logger
from octoblob import config_reader
from octoblob import diagnostics_tools
from octoblob import plotting_functions as opf
import json

class DataSource:
    """An object that supplies raw OCT data from UNP files and also digests associated
    XML files specifying acquisition parameters."""
    def __init__(self,filename,n_skip=0,x1=None,x2=None):
        cfg_filename = filename.replace('.unp','')+'.xml'
        cfg = config_reader.get_configuration(cfg_filename)

        self.cfg = cfg
        self.dtype = np.uint16
        self.n_vol = cfg['n_vol']
        self.n_slow = cfg['n_slow']
        self.n_fast = cfg['n_fast']
        self.n_depth = cfg['n_depth']
        self.n_repeats = cfg['n_bm_scans']

        if x1 is None:
            self.x1 = 0
        else:
            self.x1 = x1
            
        if x2 is None:
            self.x2 = self.n_fast
        else:
            self.x2 = x2
        
        self.bytes_per_pixel = self.dtype(1).itemsize

        self.n_bytes = self.n_vol*self.n_slow*self.n_fast*self.n_depth*self.bytes_per_pixel
        self.n_total_frames = self.n_vol*self.n_slow
        self.current_frame_index = 0
        self.filename = filename
        self.bit_shift_right = 4
        self.n_skip = n_skip

        self.saturation_value = np.iinfo(self.dtype).max

        file_size = os.stat(self.filename).st_size
        skip_bytes = self.n_skip*self.n_depth*self.bytes_per_pixel

            
        
        self.diagnostics = diagnostics_tools.Diagnostics(self.filename)
        
        try:
            assert file_size==self.n_bytes
            print('Data source established:')
            self.log_info()
            print()
            
        except AssertionError as ae:
            print('File size incorrect.\n%d\texpected\n%d\tactual'%(self.n_bytes,file_size))
            self.log_info()

    def has_more_frames(self):
        return self.current_frame_index<self.n_total_frames

    def next_frame(self,diagnostics=False):
        frame = self.get_frame(self.current_frame_index,diagnostics=diagnostics)
        self.current_frame_index+=1
        return frame


    def get_samples(self,n):
        """Get n equally spaced samples from this data set."""
        samples = []
        stride = self.n_total_frames//n
        for k in range(0,self.n_total_frames,stride):
            frame = self.get_frame(k)
            samples.append(frame)
        return samples
            
    def log_info(self):
        logging.info(self.get_info())

    def get_info(self,spaces=False):
        temp = '\nn_vol\t\t%d\nn_slow\t\t%d\nn_repeats\t%d\nn_fast\t\t%d\nn_depth\t\t%d\nbytes_per_pixel\t%d\ntotal_expected_size\t%d'%(self.n_vol,self.n_slow,self.n_repeats,self.n_fast,self.n_depth,self.bytes_per_pixel,self.n_bytes)
        if spaces:
            temp = temp.replace('\t',' ')
        return temp
    
    def get_frame(self,frame_index,volume_index=0,diagnostics=False):
        '''Get a raw frame from a UNP file. This function will
        try to read configuration details from a UNP file with
        the same name but .xml extension instead of .unp.
        Parameters:
            frame_index: the index of the desired frame; must
              include skipped volumes if file contains multiple
              volumes, unless volume_index is provided
        Returns:
            a 2D numpy array of size n_depth x n_fast
        '''
        frame = None
        # open the file and read in the b-scan
        with open(self.filename,'rb') as fid:
            # Identify the position (in bytes) corresponding to the start of the
            # desired frame; maintain volume_index for compatibility with functional
            # OCT experiments, which have multiple volumes.
            position = volume_index * self.n_depth * self.n_fast * self.n_slow * self.bytes_per_pixel + frame_index * self.n_depth * self.n_fast * self.bytes_per_pixel + self.n_skip * self.n_depth * self.bytes_per_pixel
            
            # Skip to the desired position for reading.
            fid.seek(position,0)

            # Use numpy fromfile to read raw data.
            frame = np.fromfile(fid,dtype=self.dtype,count=self.n_depth*self.n_fast)

        if frame.max()>=self.saturation_value:
            if diagnostics:
                satfig = plt.figure(figsize=(opf.IPSP,opf.IPSP),dpi=opf.screen_dpi)
                plt.hist(frame,bins=100)
                plt.title('Frame saturated with pixels >= %d.'%self.saturation_value)
                self.diagnostics.save(satfig,'saturated',frame_index)

            logging.info('Frame saturated, with pixels >= %d.'%self.saturation_value)

        if diagnostics:
            bitshiftfig = plt.figure(figsize=(opf.IPSP,2*opf.IPSP),dpi=opf.screen_dpi)
            bitshiftax1,bitshiftax2 = bitshiftfig.subplots(2,1)

            bitshiftax1.hist(frame,bins=100)
            bitshiftax1.set_title('before %d bit shift'%self.bit_shift_right)

        # Bit-shift if necessary, e.g. for Axsun/Alazar data
        if self.bit_shift_right:
            frame = np.right_shift(frame,self.bit_shift_right)

        if diagnostics:
            bitshiftax2.hist(frame,bins=100)
            bitshiftax2.set_title('after %d bit shift'%self.bit_shift_right)
            self.diagnostics.save(bitshiftfig,'bit_shift',frame_index)

        # Reshape into the k*x 2D array
        frame = frame.reshape(self.n_fast,self.n_depth).T
        frame = frame.astype(float)
        frame = frame[:,self.x1:self.x2]
        return frame


class DataSourceOptopol:
    """An object that supplies raw OCT data from Optopol files and also digests associated
    XML files specifying acquisition parameters."""
    def __init__(self,filename,n_skip=0):

        dims = np.fromfile(filename,dtype=np.int32,count=3)
        self.n_depth,self.n_fast,self.n_slow = dims
        self.n_vol = 1
        self.n_repeats = 1
        self.dtype = np.int32
        self.bytes_per_pixel = self.dtype(1).itemsize
        
        self.n_bytes = self.n_vol*self.n_slow*self.n_fast*self.n_depth*self.bytes_per_pixel
        self.n_total_frames = self.n_vol*self.n_slow
        self.current_frame_index = 0
        self.filename = filename
        self.bit_shift_right = 4
        self.n_skip = n_skip

        self.saturation_value = np.iinfo(self.dtype).max

        file_size = os.stat(self.filename).st_size
        skip_bytes = self.n_skip*self.n_depth*self.bytes_per_pixel

        self.diagnostics = diagnostics_tools.Diagnostics(self.filename)
        
        try:
            assert file_size==self.n_bytes+32*3 # include 3 32-bit header integers
            print('Data source established:')
            self.log_info()
            print()
            
        except AssertionError as ae:
            print('File size incorrect.\n%d\texpected\n%d\tactual'%(self.n_bytes,file_size))
            self.log_info()

    def has_more_frames(self):
        return self.current_frame_index<self.n_total_frames

    def next_frame(self,diagnostics=False):
        frame = self.get_frame(self.current_frame_index,diagnostics=diagnostics)
        self.current_frame_index+=1
        return frame

    def get_samples(self,n):
        """Get n equally spaced samples from this data set."""
        samples = []
        stride = self.n_total_frames//n
        for k in range(0,self.n_total_frames,stride):
            samples.append(self.get_frame(k))
        return samples
            
    def log_info(self):
        logging.info(self.get_info())

    def get_info(self,spaces=False):
        temp = '\nn_vol\t\t%d\nn_slow\t\t%d\nn_repeats\t%d\nn_fast\t\t%d\nn_depth\t\t%d\nbytes_per_pixel\t%d\ntotal_expected_size\t%d'%(self.n_vol,self.n_slow,self.n_repeats,self.n_fast,self.n_depth,self.bytes_per_pixel,self.n_bytes)
        if spaces:
            temp = temp.replace('\t',' ')
        return temp
    
    def get_frame(self,frame_index,volume_index=0,diagnostics=False):
        frame = None
        return frame

    

if __name__=='__main__':

    df = DataSource('./data/test_1.unp')
    frame = df.next_frame(diagnostics=True)
    
