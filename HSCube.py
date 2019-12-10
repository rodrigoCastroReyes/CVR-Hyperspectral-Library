from HSImageWorker import *
from scipy import signal

class HSCube(object):

	def __init__(self,dir_input=None,type_cube='raw',geometric_calibrator=None,segmentation_strategy=None):
		super(HSCube, self).__init__()
		self.type_cube = type_cube
		self.dir_root = dir_input
		
		self.geometric_calibrator = geometric_calibrator
		self.segmentator = segmentation_strategy

		self.data = None

		self.dir_input = self.set_dir_input(type_cube)
		self.load_data()

	def set_dir_input(self,type_cube):
		if type_cube == 'raw':
			return os.path.join(self.dir_root,'cubo')
		if type_cube == 'normalizado':
			return os.path.join(self.dir_root,'cuboNormalizado')
		if type_cube == 'polder':
			return os.path.join(self.dir_root,'cuboNormalizadoPolder')
		if type_cube == 'snv':
			return os.path.join(self.dir_root,'cuboNormalizadoSNV')
		if type_cube == 'moh':
			return os.path.join(self.dir_root,'cuboNormalizadoMohamed')

	def load_data(self):
		self.image_worker = HSImageWorker(self.dir_input)
		self.data = self.image_worker.load()
		if np.any(self.data):
			rows,cols,wv = self.data.shape
			self.rows = rows
			self.cols = cols
			self.num_pixels = self.rows * self.cols

	def do_geometric_calibration(self):
		self.data = self.geometric_calibrator.calibrate(self.data)

	def get_mask(self):
		if os.path.isfile(join(self.dir_root,"mask.tif")):
			print("ya existe")
			img = tiff.imread(join(self.dir_root,"mask.tif"))
			img = 1.0*img/img.max()
			return img
		mask = self.segmetator.thresholding(self.mask)
		return mask

	def get_wavelength(self,index,bin_spectral=2):
		return 0.000022*index*index*bin_spectral + 0.586*index*bin_spectral + 386.829

	def get_wavelengths(self):
		return [ self.get_wavelength(i) for i in range(520)]

	def get_spectrum_by_pixel(self,px,py,filter=False,window_size=7,order=3):
		spectral_range = self.data[py,px,:]
		if filter:
			return signal.savgol_filter(spectral_range,window_size,order)
		return spectral_range

	def get_spectrum_by_region(self,px,py,rx,ry,filter=False,window_size=11,order=3):
		region = self.data[py:py+ry,px:px+rx,:]
		rows,cols,n_wv = region.shape
		shape = (rows*cols,n_wv)#matrix size : num of spectral ranges vs num wavelengths
		matrix = region.reshape(shape)
		spectral_range = np.mean(matrix, axis=0)
		if filter:
			return signal.savgol_filter(spectral_range,window_size,order)
		return spectral_range

	def get_red(self):
		red = np.mean(self.data[:,:,182:307],axis=2)
		return red

	def get_ir(self):
		ir = np.mean(self.data[:,:,315:],axis=2)
		return ir

	def get_blue(self):
		blue = np.mean(self.data[:,:,12:97],axis=2)
		return blue

	def get_green(self):
		green = np.mean(self.data[:,:,98:181],axis=2)
		return green

	def build_rgb_components(self):
		util = ImageUtils()
		green = util.scale(self.get_green())
		red = util.scale(self.get_red())
		blue = util.scale(self.get_blue())
		bgr = [red, green, blue]
		return bgr

	def get_rgb(self):
		try:
			bgr_img = self.image_worker.read_image(join(join(self.dir_root,'fotoThor'),'output.tif'))
		except Exception as e:
			bgr_img = None
			print("Rgb Image cannot find")
		
		if bgr_img!= None:
			return bgr_img
		
		bgr = self.build_rgb_components()
		bgr_img = cv2.merge(bgr)
		return bgr_img

	def rgri(self):
		index = np.sum(self.data[:,:,181:265],axis=2)/np.sum(self.data[:,:,97:181],axis=2)
		index = index[~np.isnan(index)]
		return index[0:self.num_pixels]
	
	def pri(self):
		r_570 = np.mean(self.data[:,:,147:160],axis=2)
		r_531 = np.mean(self.data[:,:,117:139],axis=2)#525,550
		index = 1.0*(r_531-r_570)/(r_531+r_570)
		index = index[~np.isnan(index)]
		return index[0:self.num_pixels]

	def wi(self):
		r_900 = np.mean(self.data[:,:,398:439],axis=2)
		r_970 = np.mean(self.data[:,:,485:493],axis=2)
		index = 1.0*r_900/r_970
		index = index[~np.isnan(index)]

		return index[0:self.num_pixels]

	def tcari(self):
		r_700 = np.mean(self.data[:,:,248:290],axis=2)#680,730
		r_670 = np.mean(self.data[:,:,223:256],axis=2)#650,690
		r_550 = np.mean(self.data[:,:,130:148],axis=2)#540,560
		index =  3*((r_700 - r_670) - 0.2*(r_700-r_550))*(r_700/r_670)
		index = index[~np.isnan(index)]
		return index[0:self.num_pixels]