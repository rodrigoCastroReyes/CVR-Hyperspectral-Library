from scipy import signal
from sklearn import preprocessing
from scipy import stats
import itertools
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt

class HSSegmentation(object):
	"""docstring for HSGeometricCalibration"""
	def __init__(self):
		super(HSGeometricCalibration, self).__init__()

	def thresholding(self,data):
		pass

class DIPSegmentation(HSSegmentation):
	"""docstring for ClassName"""
	def __init__(self):
		HSSegmentation.__init__(self)

	def thresholding(self,data):
		return None
		
class MLSegmentation(HSSegmentation):
	"""docstring for MLStrategy"""
	def __init__(self, arg):
		HSSegmentation.__init__(self)
	
	def thresholding(self,data):

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
		#tiff.imsave("result.tif",result)

		base_mask = BaseMask(result)
		dilation = Erosion(base_mask,size_selem=3,n_operations=1)
		mask = dilation.transform()

		return mask