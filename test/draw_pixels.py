import sys
sys.path.append("../")
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import zscore
from matplotlib.legend_handler import HandlerLine2D
from HSCube import *
from os.path import join

import math

def onclick(event):
	if event.xdata != None and event.ydata != None:
		c = int(math.floor(event.xdata))#columna
		r = int(math.floor(event.ydata))#fila
		print(r,c)
		spectrum = cube.get_spectrum_by_pixel(c,r,filter=False)
		print(spectrum)
		#spectrum = 1.0*spectrum/np.sum(spectrum[100:400])
		img_ = np.copy(img)
		img[r,c,:] = [255,0,0]
		axes[0].imshow(img,interpolation="nearest")
		axes[0].set_axis_off()
		#axes[1].cla()
		axes[1].plot(spectral_domain,spectrum)
		axes[1].set_xlabel('wavelength')
		axes[1].set_ylabel('R')

		fig.canvas.draw()

dir_input = sys.argv[1]
cube = HSCube(dir_input,type_cube='normalized')
spectral_domain = cube.get_wavelengths()
if cube.load_data() :
	print("Loading HSCube")
	img = cube.get_rgb()
	print(img.shape)
	#tvi = cube.tvi()
	#tvi = tvi.reshape((cube.rows,cube.cols))

	fig, axes = plt.subplots(1,2,figsize=(10, 6))
	axes[0].imshow(img,interpolation="nearest")
	axes[0].set_axis_off()
	fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
