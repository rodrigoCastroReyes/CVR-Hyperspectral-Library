import os
import numpy as np
from HSImageWorker import *
from ImageMask import *
import cv2
from skimage import filters
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
from scipy import signal

class HSCube(object):
	"""docstring for HSCube"""
	def __init__(self,dir_input=None,type_cube='normalizado',spectral_range_flag='all'):
		super(HSCube, self).__init__()
		self.type_cube = type_cube
		self.dir_root = dir_input
		self.leaf = None
		self.background = None
		self.mask = None
		if dir_input == None:
			return			
		if self.type_cube == 'raw':
			self.dir_input = os.path.join(dir_input,'cubo')
		if self.type_cube == 'normalizado':
			self.dir_input = os.path.join(dir_input,'cuboNormalizado')
		if self.type_cube == 'polder':
			self.dir_input = os.path.join(dir_input,'cuboNormalizadoPolder')
		if self.type_cube == 'snv':
			self.dir_input = os.path.join(dir_input,'cuboNormalizadoSNV')

		self.load_data(spectral_range_flag)
		
	def load_data(self,spectral_range_flag):
		self.image_worker = HSImageWorker(self.dir_input,spectral_range=spectral_range_flag)
		self.data = self.image_worker.load()
		if np.any(self.data):
			rows,cols,wv = self.data.shape
			self.rows = rows
			self.cols = cols
			self.num_pixels = self.rows * self.cols

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
			print ("Rgb Image cannot find")
		if bgr_img!= None:
			return bgr_img
		
		bgr = self.build_rgb_components()
		bgr_img = cv2.merge(bgr)
		return bgr_img

	def get_background(self,binary_not=True):
		dir_input = os.path.join(self.dir_root,'cubo')
		image_worker_helper = HSImageWorker(dir_input)
		data = image_worker_helper.load()
	
		red = np.mean(data[:,:,182:307],axis=2)
		ir = np.mean(data[:,:,315:],axis=2)

		svi_image = ir

		base_mask = BaseMask(svi_image)
		thresholding = Thresholding(base_mask)
		morph = Closing(thresholding,size_selem=6)
		filling = Filling(morph)
		erosion = Erosion(filling)
		background = erosion.transform()
		tiff.imsave("mask.tif",background)

		if binary_not:
			background = np.uint8(255*np.logical_not(background))
		
		self.background = background

		return background

	def get_leaf(self,binary_not=False):
		dir_input = os.path.join(self.dir_root,'cubo')
		image_worker_helper = HSImageWorker(dir_input)
		data = image_worker_helper.load()
		ir = np.mean(data[:,:,315:],axis=2)

		base_mask = BaseMask(ir)
		hessian = PCBR(base_mask,scale=2.0)
		thresholding = Thresholding(hessian)
		morph = Closing(thresholding,size_selem=6)
		dilation = Dilation(morph)
		skeleton = Skeletonization(dilation)
		mask = skeleton.transform()
		if binary_not:
			background = np.uint8(255*np.logical_not(background))
		return mask


	def get_sides(self):
		mask = self.get_mask().flatten()

		if self.type_cube != 'normalizado':
			dir_input = os.path.join(self.dir_root,'cuboNormalizado')
			image_worker_helper = HSImageWorker(dir_input)
			new_data = image_worker_helper.load()
			data = new_data.reshape((self.rows*self.cols,520))
		else:
			data = self.data.reshape((self.rows*self.cols,520))
		
		positions = [ [row,col] for row,col in itertools.product(range(self.rows),range(self.cols))]
		data = np.column_stack((positions,mask,data))
		features = ['px','py','no_background']
		for i in range(1,521):
			features.append( "l" + str(i))
		df = pd.DataFrame(data = data,columns=features)
		df_ = df[df['no_background'] == 1]

		pca = PCA(n_components=2,svd_solver='full')
		data_r = pca.fit_transform(df_[df_.columns[150:400]])
		clustering = KMeans(n_clusters=2, random_state=0).fit(data_r)
		#clustering = DBSCAN(eps=0.5, min_samples=5).fit(data_r)
		classes = clustering.labels_
		sides = np.zeros((self.rows,self.cols))

		for i,(index,row) in enumerate(df_.iterrows()):
			r = int(row["px"])
			c = int(row["py"])
			label = classes[i]
			if label != -1:
				sides[r,c] = label

		colors = {
			0: '#00ff00',
			1: '#ff0000',
			2: '#0000ff',
			3: '#000fff',
			-1: '#00ffff'
		}
		labels = [ colors[label] for label in classes]
		"""
		fig, axes = plt.subplots(1,1,figsize=(10, 6))
		axes.scatter(data_r[:,0],data_r[:,1],c=labels)
		axes.set_xlabel('1st Component')
		axes.set_ylabel('2nd Component')
		plt.show()
		"""

		return sides.flatten()

	def get_mask(self):
		
		if os.path.isfile(join(self.dir_root,"mask.tif")):
			img = tiff.imread(join(self.dir_root,"mask.tif"))
			img = 1.0*img/img.max()
			return img
		if self.type_cube != 'raw':
			dir_input = os.path.join(self.dir_root,'cubo')
			image_worker_helper = HSImageWorker(dir_input)
			new_data = image_worker_helper.load()
			data = new_data.reshape((self.rows*self.cols,520))
		else:
			data = self.data.reshape((self.rows*self.cols,520))
		
		pca = PCA(n_components=4)
		data_r = pca.fit_transform(data[:,100:400])

		kmeans = KMeans(n_clusters=4, random_state=0).fit(data_r)
		labels = kmeans.labels_
		plt.scatter(data_r[:,0],data_r[:,1],c=labels)
		plt.show()
		
		labels_img = labels.reshape((self.rows,self.cols))
		fig, axes = plt.subplots(1,1,figsize=(16, 6))
		axes.imshow(labels_img, interpolation='nearest', cmap=plt.cm.gray)
		axes.set_axis_off()
		plt.show()

		centroids = []
		for i in range(0,4):
			cluster = data[np.where(labels == i)]
			centroids.append(np.mean(cluster))
		indexes = np.argsort(centroids)[::-1]
		leaf_labels = indexes[0:2]
		background_labels = indexes[2:]
		result = np.zeros((self.rows,self.cols),dtype=np.uint8)

		for y,x in itertools.product(range(self.rows),range(self.cols)):
			if labels_img[y,x] in leaf_labels:
				result[y,x] = 1
			elif labels_img[y,x] in background_labels:
				result[y,x] = 0
		
		#result = np.uint8(255*result.shape)
		tiff.imsave("result.tif",result)
	
		base_mask = BaseMask(result)
		dilation = Erosion(base_mask,size_selem=3,n_operations=1)
		mask = dilation.transform()

		return mask

	def get_rectanlge_of_leaf(self):
		if not(self.background):
			self.background = self.get_background(False)
		regions = measure.regionprops(self.background)
		region = regions[0]
		return region.bbox

	def get_side_of_leaf(self,px,py,rx,ry):
		if not(self.background.any()):
			self.background = self.get_background()
		regions = measure.regionprops(self.background)
		region = regions[0]
		c_py,c_px = region.centroid
		cols = range(px,px+rx)
		rows = range(py,py+ry)
		
		dist_col = []
		
		for row,col in itertools.product(rows,cols):
			dist_col.append(c_px - col)
		
		dist_col = np.mean(dist_col,axis=0)

		if dist_col < 0 :
			return 'derecha'
		else:
			return 'izquierda'

	def rgri(self):
		index =np.sum(self.data[:,:,181:265],axis=2)/np.sum(self.data[:,:,97:181],axis=2)
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

	def tvi(self):
		r_750 = np.mean(self.data[:,:,290:332],axis=2)#730,780
		r_550 = np.mean(self.data[:,:,130:148],axis=2)#540,560
		r_670 = np.mean(self.data[:,:,223:256],axis=2)#650,690

		index = 0.5*(120*(r_750 - r_550) - 200*(r_670-r_550))
		index = index[~np.isnan(index)]

		return index[0:self.num_pixels]

	def ir(self):
		ir = np.mean(self.data[:,:,315:],axis=2)
		ir = ir[~np.isnan(ir)]
		return ir[0:self.num_pixels]

class HSSyntheticCube(HSCube):
	"""docstring for HSSyntheticCube"""
	def __init__(self,raw_data):
		HSCube.__init__(self,dir_input=None)
		self.build(raw_data)

	def build(self,raw_data):
		real_num_pixels,wavelenghts = raw_data.shape
		mod = real_num_pixels%10
		num_pixels,wavelenghts = raw_data.shape
		self.num_pixels = num_pixels

		if mod != 0:
			new_data = np.array([ 1 for i in range(520)])
			for i in range(10-mod):
				self.data = np.vstack((raw_data,new_data))

		num_pixels,wavelenghts = raw_data.shape
		rows = num_pixels/10
		cols = 10
		self.data = np.reshape(raw_data,(rows,cols,wavelenghts))
		self.rows = rows
		self.cols = cols
