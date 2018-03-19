import pandas as pd
import sys
import tifffile as tiff
sys.path.append("/home/rodfcast/Documents/CVR/Proyecto RIP/Hyperspectral_Library")
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from HSCube import *
from ImageWorker import *
from ImageUtils import *
from scipy import stats
from skimage import measure
from skimage.measure import label
import itertools

dir_input = sys.argv[1]
type_norms = ['normalizado','polder','snv','moh']
imgs_labels = []
for type_norm in type_norms:
	hype_cube = HSCube(dir_input=dir_input,type_cube = type_norm)
	rows = hype_cube.rows
	cols = hype_cube.cols

	rgri = hype_cube.rgri()
	pri = hype_cube.pri()
	wi = hype_cube.wi()
	tcari = hype_cube.tcari()

	data = np.column_stack((rgri,pri,wi,tcari))
	data = np.array(data)

	pca = PCA(n_components=2)
	data_r = pca.fit_transform(data)

	kmeans = KMeans(n_clusters=5).fit(data_r)
	centroids = kmeans.cluster_centers_
	labels = kmeans.labels_
	imgs = labels.reshape((rows,cols))
	imgs_labels.append(imgs)

fig, ax = plt.subplots(figsize=(10, 6),nrows=2, ncols=2)
fig.suptitle("Unsupervised Classification", fontsize=16)
for k,(i,j) in enumerate(itertools.product(range(2),range(2))):
	ax[i,j].set_axis_off()
	ax[i,j].imshow(imgs_labels[k])
plt.show()