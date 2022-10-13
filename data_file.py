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
import config_reader

class DataFile:

    def __init__(self,filename,n_skip=0):
        
        cfg_filename = filename.replace('.unp','')+'.xml'
        cfg = config_reader.get_configuration(cfg_filename)

        self.cfg = cfg
        self.dtype = np.uint16
        self.n_vol = cfg['n_vol']
        self.n_slow = cfg['n_slow']
        self.n_fast = cfg['n_fast']
        self.n_depth = cfg['n_depth']
        self.n_repeats = cfg['n_bm_scans']

        print(dir(self.dtype(1)))
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
        
        try:
            assert file_size==self.n_bytes
            print('Data source established:')
            self.print_volume_info()
            print()
            
        except AssertionError as ae:
            print('File size incorrect.\n%d\texpected\n%d\tactual'%(self.n_bytes,file_size))
            self.print_volume_info()

    def has_more_frames(self):
        return self.current_frame_index<self.n_total_frames

    def next_frame(self):
        frame = self.get_frame(self.current_frame_index)
        self.current_frame_index+=1
        return frame
            
    def print_volume_info(self):
        #print('n_vol\t\t%d\nn_slow\t\t%d\nn_repeats\t%d\nn_fast\t\t%d\nn_depth\t\t%d\nbytes_per_pixel\t%d\ntotal_expected_size\t%d'%(self.n_vol,self.n_slow,self.n_repeats,self.n_fast,self.n_depth,self.bytes_per_pixel,self.n_bytes))
        print(self.get_info())

        
    def get_info(self,spaces=False):
        temp = 'n_vol\t\t%d\nn_slow\t\t%d\nn_repeats\t%d\nn_fast\t\t%d\nn_depth\t\t%d\nbytes_per_pixel\t%d\ntotal_expected_size\t%d'%(self.n_vol,self.n_slow,self.n_repeats,self.n_fast,self.n_depth,self.bytes_per_pixel,self.n_bytes)
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
                    plt.figure(figsize=(opf.IPSP,opf.IPSP),dpi=opf.screen_dpi)
                    plt.hist(frame,bins=100)
                    plt.title('Frame saturated with pixels >= %d.'%self.saturation_value)
                print('Frame saturated, with pixels >= %d.'%self.saturation_value)
            
            if diagnostics:
                plt.figure(figsize=(opf.IPSP,2*opf.IPSP),dpi=opf.screen_dpi)
                plt.subplot(2,1,1)
                plt.hist(frame,bins=100)
                plt.title('before %d bit shift'%self.bit_shift_right)
                
            # Bit-shift if necessary, e.g. for Axsun/Alazar data
            if self.bit_shift_right:
                frame = np.right_shift(frame,self.bit_shift_right)

            if diagnostics:
                plt.subplot(2,1,2)
                plt.hist(frame,bins=100)
                plt.title('after %d bit shift'%self.bit_shift_right)
                
                
            # Reshape into the k*x 2D array
            frame = frame.reshape(self.n_fast,self.n_depth).T


            if diagnostics:
                plt.figure(figsize=(opf.IPSP,opf.IPSP),dpi=opf.screen_dpi)
                plt.imshow(frame,aspect='auto',interpolation='none',cmap='gray')
                plt.colorbar()
                plt.title('raw data (bit shifted %d bits)'%self.bit_shift_right)
                save_diagnostics(diagnostics,'raw_data')
            
        return frame



if __name__=='__main__':

    df = DataFile('./data/test_1.unp')
    plt.imshow(df.next_frame())
    plt.show()
