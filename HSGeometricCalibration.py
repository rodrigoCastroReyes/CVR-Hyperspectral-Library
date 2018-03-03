from scipy import signal
from sklearn import preprocessing
from scipy import stats
import itertools
import numpy as np
import os

class HSGeometricCalibration(object):
	"""docstring for HSGeometricCalibration"""
	def __init__(self):
		super(HSGeometricCalibration, self).__init__()

	def calibrate(self,data):
		pass

class Polder(HSGeometricCalibration):
	"""docstring for HSGeometricCalibration"""
	def __init__(self,):
		HSGeometricCalibration.__init__(self)
		self.calibration_type = "cuboNormalizadoPolder"

	def calibrate(self,data):
		y_spatial,x_spatial,w_spectral = data.shape
		new_cube = np.zeros((y_spatial,x_spatial,w_spectral))
		for y,x in itertools.product(range(y_spatial),range(x_spatial)):
			new_cube[y,x,:] = (1.0*data[y,x,:])/np.sum(data[y,x,:])
		return new_cube


class SNV(HSGeometricCalibration):
	"""docstring for HSSNV"""
	def __init__(self):
		HSGeometricCalibration.__init__(self)
		self.calibration_type = "cuboNormalizadoSNV"

	def calibrate(self,data):
		y_spatial,x_spatial,w_spectral = data.shape
		new_cube = np.zeros((y_spatial,x_spatial,w_spectral))
		for y,x in itertools.product(range(y_spatial),range(x_spatial)):
			new_cube[y,x,:] = preprocessing.scale(1.0*data[y,x,:])
		return new_cube

class MOL(HSGeometricCalibration):
	"""docstring for HS"""
	def __init__(self,y_ref,x_ref,rx,ry,mask):
		HSGeometricCalibration.__init__(self)
		self.calibration_type = "cuboNormalizadoMOL"
		self.y_ref = y_ref
		self.x_ref = x_ref
		self.rx = rx
		self.ry = ry
		self.img_mask = mask

	def get_ref_spectrum(self,data):
		spectral_range_ref = np.mean(cube[self.y_ref:(self.y_ref+self.ry),self.x_ref:(self.x_ref+self.rx),:],axis=(0,1))
		return spectral_range_ref

	def calibrate(self,data):
		spectral_range_ref = self.get_ref_spectrum(data)
		y_spatial,x_spatial,w_spectral = data.shape
		new_cube = np.zeros((y_spatial,x_spatial,w_spectral))
		for y,x in itertools.product(range(y_spatial),range(x_spatial)):
			no_is_backgroud = self.img_mask[y,x]
			if no_is_backgroud :
				beta, alpha, r_value, p_value, std_err = stats.linregress(np.matrix(spectral_range_ref),np.matrix(data[y,x,:]))
				new_cube[y,x,:] = 1.0*(data[y,x,:] - alpha )/beta
		return new_cube