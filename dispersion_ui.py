#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as spo
import sys,os,glob
import scipy.interpolate as spi

#######################################
## Constants here--adjust as necessary:

dB_lims = [45,85]
crop_height = 300 # height of viewable B-scan, centered at image z centroid (center of mass)

# step sizes for incrementing/decrementing coefficients:
mapping_steps = [1e-9,1e-6]
dispersion_steps = [1e-9,1e-8]
fbg_position = 90
bit_shift_right = 4
window_sigma = 0.9

ui_width = 400
ui_height = 600
#######################################

# Now we'll define some functions for the half-dozen or so processing
# steps:

def load_spectra(fn,index=50):
    ext = os.path.splitext(fn)[1]
    if ext.lower()=='.unp':
        from octoblob import config_reader,data_source
        import octoblob as blob
        
        src = data_source.DataSource(fn)

        index = index%(src.n_slow*src.n_vol)
        spectra = src.get_frame(index)
    elif ext.lower()=='.npy':
        spectra = np.load(fn)
    else:
        sys.exit('File %s is of unknown type.'%fn)
    return spectra

# We need a way to estimate and remove DC:
def dc_subtract(spectra):
    """Estimate DC by averaging spectra spatially (dimension 1),
    then subtract by broadcasting."""
    dc = spectra.mean(1)
    # Do the subtraction by array broadcasting, for efficiency.
    # See: https://numpy.org/doc/stable/user/basics.broadcasting.html
    out = (spectra.T-dc).T
    return out


# Next we need a way to adjust the values of k at each sample, and then
# interpolate into uniformly sampled k:
def k_resample(spectra,coefficients):
    """Resample the spectrum such that it is uniform w/r/t k.
    Notes:
      1. The coefficients here are for a polynomial defined on
         pixels, so they're physically meaningless. It would be
         better to define our polynomials on k, because then
         we could more easily quantify and compare the chirps
         of multiple light sources, for instance. Ditto for the
         dispersion compensation code.
    """
    coefficients = coefficients + [0.0,0.0]
    # x_in specified on array index 1..N+1
    x_in = np.arange(1,spectra.shape[0]+1)

    # define an error polynomial, using the passed coefficients, and then
    # use this polynomial to define the error at each index 1..N+1
    error = np.polyval(coefficients,x_in)
    x_out = x_in + error

    # using the spectra measured at indices x_in, interpolate the spectra at indices x_out
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    interpolator = spi.interp1d(x_in,spectra,axis=0,kind='cubic',fill_value='extrapolate')
    interpolated = interpolator(x_out)
    return interpolated

# Next we need to dispersion compensate; for historical reasons the correction polynomial
# is defined on index x rather than k, but for physically meaningful numbers we should
# use k instead
def dispersion_compensate(spectra,coefficients):
    coefs = list(coefficients) + [0.0,0.0]
    # define index x:
    x = np.arange(1,spectra.shape[0]+1)
    # define the phasor and multiply by spectra using broadcasting:
    dechirping_phasor = np.exp(-1j*np.polyval(coefs,x))
    dechirped = (spectra.T*dechirping_phasor).T
    return dechirped


# Next we multiply the spectra by a Gaussian window, in order to reduce ringing
# in the B-scan due to edges in the spectra:
def gaussian_window(spectra,sigma):
    # Define a Gaussian window with passed sigma
    x = np.exp(-((np.linspace(-1.0,1.0,spectra.shape[0]))**2/sigma**2))
    # Multiply spectra by window using broadcasting:
    out = (spectra.T*x).T
    return out


# Now let's define a processing function that takes the spectra and two dispersion coefficients
# and produces a B-scan:
def process_bscan(spectra,mapping_coefficients=[0.0],dispersion_coefficients=[0.0],window_sigma=0.9):
    spectra = dc_subtract(spectra)
    # When we call dispersion_compensate, we have to pass the c3 and c2 coefficients as well as
    # two 0.0 values, to make clear that we want orders 3, 2, 1, 0. This enables us to use the
    # polyval function of numpy instead of writing the polynomial ourselves, e.g. c3*x**3+c2*x**x**2,
    # since the latter is more likely to cause bugs.
    spectra = k_resample(spectra,mapping_coefficients)
    spectra = dispersion_compensate(spectra,dispersion_coefficients)
    spectra = gaussian_window(spectra,sigma=window_sigma)
    bscan = np.fft.fft(spectra,axis=0)
    return bscan



# An example of optimizing dispersion:

# First, we need an objective function that takes the two dispersion coefficients and outputs
# a single value to be minimized; for simplicity, we'll use the reciprocal of the brightest
# pixel in the image. An oddity here is that the function can see outside its scope and thus
# has access to the variable 'spectra', defined at the top by loading from the NPY file. We
# then call our process_bscans function, using the coefficients passed into this objective
# function. From the resulting B-scan, we calculate our value to be minimized:
def obj_func(coefs,save=False):
    bscan = process_bscan(spectra,coefs)
    # we don't need the complex conjugate, so let's determine the size of the B-scan and crop
    # the bottom half (sz//2:) for use. (// means integer division--we can't index with floats;
    # also, the sz//2: is implied indexing to the bottom of the B-scan:
    sz,sx = bscan.shape
    bscan = bscan[sz//2:,:]
    # we also want to avoid DC artifacts from dominating the image brightness or gradients,
    # so let's remove the bottom, using negative indexing.
    # See: https://numpy.org/devdocs/user/basics.indexing.html
    bscan = bscan[:-50,:]
    # Finally let's compute the amplitude (modulus) max and return its reciprocal:
    bscan = np.abs(bscan)
    bscan = bscan[-300:] # IMPORTANT--THIS WON'T WORK IN GENERAL, ONLY ON THIS DATA SET 16_53_25
    out = 1.0/np.max(bscan)
    
    # Maybe we want to visualize it; change to False to speed things up
    if True:
        # clear the current axis
        plt.cla()
        # show the image:
        plt.imshow(20*np.log10(bscan),cmap='gray',clim=dB_lims)
        # pause:
        plt.pause(0.001)

    if save:
        order = len(coefs)+1
        os.makedirs('dispersion_compensation_results',exist_ok=True)
        plt.cla()
        plt.imshow(20*np.log10(bscan),cmap='gray',clim=dB_lims)
        plt.title('order %d\n %s'%(order,list(coefs)+[0.0,0.0]),fontsize=10)
        plt.colorbar()
        plt.savefig('dispersion_compensation_results/order_%d.png'%order,dpi=150)
    return out


# Now we can define some settings for the optimization:

def optimize_dispersion(spectra,obj_func,initial_guess):

    # spo.minimize accepts an additional argument, a dictionary containing further
    # options; we want can specify an error tolerance, say about 1% of the bounds.
    # we can also specify maximum iterations:
    optimization_options = {'xatol':1e-10,'maxiter':10000}

    # optimization algorithm:
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    method = 'nelder-mead'

    # Now we run it; Nelder-Mead cannot use bounds, so we pass None
    res = spo.minimize(obj_func,initial_guess,method='nelder-mead',bounds=None,options=optimization_options)
    

if len(sys.argv)>=2:
    if sys.argv[1]=='0':
        spectra = np.load('./spectra_00100.npy')
        bscan = process_bscan(spectra,window_sigma=900)
        plt.imshow(20*np.log10(np.abs(bscan))[-300:-100],cmap='gray',clim=dB_lims)
        plt.show()

        sys.exit()

class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0
        self.mapping_coefficients = [0.0,0.0]
        self.dispersion_coefficients = [0.0,0.0]
        
        self.mapping_steps = mapping_steps
        self.dispersion_steps = dispersion_steps

        self.cropz1 = None
        self.cropz2 = None
        
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("Image Viewer")
        self.resize(ui_width, ui_height)

        self.spectra = None
        self.crop_half_height = crop_height//2
        self.bscan_max_gradient = 0.0
        self.bscan_max_amplitude = 0.0
        self.window_sigma = window_sigma
        self.dB_lims = dB_lims
        self.bscan_max_amplitude_original = None
        self.image_loaded = False
        self.frame_index = 0
        
        if len(sys.argv)>=2:
            print(sys.argv)
            fn = sys.argv[1]
            self.spectra = load_spectra(fn,self.frame_index)
            self.image_loaded = True
            
            self.update_image()
            self.filename = fn
            self.fitToWindowAct.setEnabled(True)

        

    def update_image(self):
        image = self.get_bscan_qimage(self.spectra,mapping_coefficients=self.mapping_coefficients,dispersion_coefficients=self.dispersion_coefficients)
        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0
        
        self.scrollArea.setVisible(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

        def f(L):
            return ['%0.3e'%k for k in L]
        
        print('mapping coefficients: %s'%(f(self.mapping_coefficients)))
        print('dispersion coefficients: %s'%(f(self.dispersion_coefficients)))
        print('dB contrast lims: %s'%self.dB_lims)
        print('spectral Gaussian window sigma: %0.3f'%self.window_sigma)
        print('frame index: %d'%self.frame_index)
        print('bscan max amplitude: %0.1f'%self.bscan_max_amplitude)
        print('bscan max gradient: %0.1f'%self.bscan_max_gradient)
        print('bscan mean gradient: %0.1f'%self.bscan_mean_gradient)
        n_hash = min(round(self.bscan_max_amplitude/self.bscan_max_amplitude_original*100.0),120)
        print('max: '+'#'*n_hash)
                
    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.npy *.unp)', options=options)
        if fileName:
            # self.spectra = np.load(fileName)
            self.spectra = load_spectra(fileName)
            self.image_loaded = True
            #self.dispersion_coefficients = [0.0,0.0]
            #self.mapping_coefficients = [0.0,0.0]
            
            self.update_image()
            self.fitToWindowAct.setEnabled(True)
            self.filename = fileName
            
    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "OCT Viewer",
                          "<p>For the following adjustments, use CTRL for 10x and SHIFT-CTRL for 100x:<br>"
                          "Y/H keys increase/decrease 3rd order mapping coefficient<br>"
                          "U/J keys increase/decrease 2rd order mapping coefficient<br>"
                          "I/K keys increase/decrease 3rd order dispersion coefficient<br>"
                          "O/L keys increase/decrease 2rd order dispersion coefficient<br></p>"
                          "<p>R/F increase/decrease lower contrast limit (dB)</p>"
                          "<p>T/G increase/decrease upper contrast limit (dB)</p>"
                          "<p>E/D increase/decrease spectral Gaussian window sigma</p>"
                          "<p>Z sets dispersion and mapping coefficients to all zeros</p>"
                          "<p>See menus for other options.</p>")

    def createActions(self):
        self.openAct = QAction("O&pen...", self, shortcut="Ctrl+P", triggered=self.open)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl+=", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About OCTView", self, triggered=self.about)
        

    def c3up(self,multiplier=1.0):
        self.dispersion_coefficients[0]+=self.dispersion_steps[0]*multiplier
        self.update_image()
    def c3down(self,multiplier=1.0):
        self.dispersion_coefficients[0]-=self.dispersion_steps[0]*multiplier
        self.update_image()
    def c2up(self,multiplier=1.0):
        self.dispersion_coefficients[1]+=self.dispersion_steps[1]*multiplier
        self.update_image()
    def c2down(self,multiplier=1.0):
        self.dispersion_coefficients[1]-=self.dispersion_steps[1]*multiplier
        self.update_image()


    def m3up(self,multiplier=1.0):
        self.mapping_coefficients[0]+=self.mapping_steps[0]*multiplier
        self.update_image()
    def m3down(self,multiplier=1.0):
        self.mapping_coefficients[0]-=self.mapping_steps[0]*multiplier
        self.update_image()
    def m2up(self,multiplier=1.0):
        self.mapping_coefficients[1]+=self.mapping_steps[1]*multiplier
        self.update_image()
    def m2down(self,multiplier=1.0):
        self.mapping_coefficients[1]-=self.mapping_steps[1]*multiplier
        self.update_image()


    def ulimup(self):
        self.dB_lims[1]+=1
        self.update_image()

    def ulimdown(self):
        self.dB_lims[1]-=1
        self.update_image()

    def llimup(self):
        self.dB_lims[0]+=1
        self.update_image()

    def llimdown(self):
        self.dB_lims[0]-=1
        self.update_image()
    

    def wsup(self):
        self.window_sigma*=1.1
        self.update_image()

    def wsdown(self):
        self.window_sigma/=1.1
        self.update_image()

    def keyPressEvent(self, e):
        if not self.image_loaded:
            return
        
        mod = e.modifiers()

        if (mod & Qt.ControlModifier) and (mod & Qt.ShiftModifier):
            multiplier = 100.0
        elif (mod & Qt.ControlModifier):
            multiplier = 10.0
        else:
            multiplier = 1.0
            
        
        # if e.modifiers() == Qt.ControlModifier:
        #     multiplier = 10.0
        # elif e.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
        #     multiplier = 100.0
        # else:
        #     multiplier = 1.0
            
        if e.key() == Qt.Key_Y:
            self.m3up(multiplier)
        elif e.key() == Qt.Key_H:
            self.m3down(multiplier)
            
        elif e.key() == Qt.Key_U:
            self.m2up(multiplier)
        elif e.key() == Qt.Key_J:
            self.m2down(multiplier)

        elif e.key() == Qt.Key_I:
            self.c3up(multiplier)
        elif e.key() == Qt.Key_K:
            self.c3down(multiplier)
            
        elif e.key() == Qt.Key_O:
            self.c2up(multiplier)
        elif e.key() == Qt.Key_L:
            self.c2down(multiplier)

        elif e.key() == Qt.Key_R:
            self.llimup()
        elif e.key() == Qt.Key_F:
            self.llimdown()

        elif e.key() == Qt.Key_T:
            self.ulimup()

        elif e.key() == Qt.Key_G:
            self.ulimdown()
            
        elif e.key() == Qt.Key_E:
            self.wsup()
        elif e.key() == Qt.Key_D:
            self.wsdown()

        elif e.key() == Qt.Key_PageUp:
            self.frame_index+=1
            self.spectra = load_spectra(self.filename,self.frame_index)
            self.update_image()

        elif e.key() == Qt.Key_PageDown:
            self.frame_index-=1
            self.spectra = load_spectra(self.filename,self.frame_index)
            self.update_image()

        elif e.key() == Qt.Key_Z:
            self.dispersion_coefficients = [0.0,0.0]
            self.mapping_coefficients = [0.0,0.0]
            self.update_image()
            
        elif e.key() == Qt.Key_Escape:
            self.close()
            

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def get_bscan_qimage(self,spectra,mapping_coefficients=[0.0],dispersion_coefficients=[0.0],window_sigma=0.9):
        bscan = process_bscan(spectra,mapping_coefficients,dispersion_coefficients,self.window_sigma)
        bscan = bscan[bscan.shape[0]//2:,:]
        bscan = np.abs(bscan)
        self.bscan_max_amplitude = np.max(bscan)
        abs_grad = np.abs(np.diff(bscan,axis=0))
        self.bscan_max_gradient = np.max(abs_grad)
        self.bscan_mean_gradient = np.mean(abs_grad)

        if self.bscan_max_amplitude_original is None:
            self.bscan_max_amplitude_original = self.bscan_max_amplitude

        bscan = 20*np.log10(bscan)
        #bscan = bscan.T


        if self.cropz1 is None:
            bprof = np.mean(bscan,axis=1)
            bprof = bprof - np.min(bprof)
            z = np.arange(len(bprof))
            com = int(round(np.sum(bprof*z)/np.sum(bprof)))
            bscan = bscan[com-self.crop_half_height:com+self.crop_half_height,:]
            self.cropz1 = com-self.crop_half_height
            self.cropz2 = com+self.crop_half_height
        else:
            bscan = bscan[self.cropz1:self.cropz2]

        bscan = np.clip(bscan,*self.dB_lims)

        bscan = (bscan-np.min(bscan))/(np.max(bscan)-np.min(bscan))
        bscan = bscan*255
        img = np.round(bscan).astype(np.uint8)

        img = np.zeros((bscan.shape[0], bscan.shape[1]), dtype=np.uint8)
        img[:] = bscan[:]
        # Turn up red channel to full scale
        #img[...,0] = 255
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)

        #qimage = QImage(bscan,bscan.shape[1],bscan.shape[0],QImage.Format_RGB888)
        return qimage


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
