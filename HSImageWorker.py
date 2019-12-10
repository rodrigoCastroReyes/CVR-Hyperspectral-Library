import os
import tifffile as tiff
import numpy as np
import re
import math

class HSImageWorker(object):
	#Hiperspectral Image Worker: lee directorio y carga imagenes en estructura de datos

	def __init__(self, dir_input=None):
		super(HSImageWorker, self).__init__()
		self.dir_input = dir_input

	def set_spectral_range(self,min_wv,max_wv):
		self.spectral_range = np.array([ i for i in range(min_wv,max_wv)])

	def set_input_directory(self,dir_input):
		self.dir_input = dir_input

	def get_input_directory(self):
		return self.dir_input

	def read_image(self,dir):
		return tiff.imread(dir)

	def get_file_names(self):
		return self.files_names

	def get_index(self,d,bin_spectral=1):
		coeff = [0.000022*bin_spectral, 0.586*bin_spectral,  386.829 - d]
		solutions = np.roots(coeff)
		x = solutions[solutions>0]
		return int(math.floor(x[0]))

	def get_file_names(self,min_index,max_index):#lee los nombres de los archivos del directorio cubo* 
		#build a tuple (wavelength,filename)
		onlyfiles = [ (float(f.split('longitud')[1].split('.tif')[0]), os.path.join(self.dir_input,f)) for f in os.listdir(self.dir_input) if os.path.isfile(os.path.join(self.dir_input, f)) and "longitud" in (f) ]
		onlyfiles = np.array(sorted(onlyfiles,key=lambda tup: tup[0]))
		first_wv = float(onlyfiles[0][0])
		last_wv = float(onlyfiles[-1][0])
		index_first = self.get_index(first_wv)
		last_first = self.get_index(last_wv)
		#self.set_spectral_range(index_first,last_first)
		#onlyfiles = onlyfiles[self.spectral_range]
		return np.array([ file for w,file in onlyfiles[min_index:max_index]])

	def load(self,min_index,max_index):#lee las imagenes desde la camara thor
		if os.path.isdir(self.dir_input) and re.search("cubo",self.dir_input)!=None:
			#images = np.array([])
			self.files_names = self.get_file_names(min_index,max_index)
			wvs = len(self.files_names)#number of wavelengths
			one_image_fn = self.files_names[0]
			test_img = np.array(self.read_image(one_image_fn))
			rows,cols = test_img.shape#spatil axis
			images = np.zeros((rows,cols,wvs))
			for k,file_name in enumerate(self.files_names):
				images[:,:,k] = np.array(self.read_image(file_name))
			return images
<<<<<<< HEAD
		print("it can't find directory")
=======
		print ("it can't find directory")
>>>>>>> 3e3e26721b52d9725d7524ac37d3f6709f89367c
		return None

	def write_images(self,imgs,dir):
		freeimg.write_multipage(imgs,'myimages.tiff')