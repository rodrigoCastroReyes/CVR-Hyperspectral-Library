import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from matplotlib.legend_handler import HandlerLine2D


class Drawer(object):
	"""docstring for Drawer"""
	def __init__(self):
		super(Drawer, self).__init__()

	def add_data(self,domain,range,label,loc,color='b',x_label = "wavelength(nm)",y_label="reflectance"):
		plt.axes()
		#circle = plt.Circle((0,0), radius=0.01, fc='y')
		#plt.gca().add_patch(circle)
		#plt.show()
		axes = plt.gca()
		#axes.set_xlim([xmin,xmax])
		#axes.set_ylim([0,0.4])
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		p1,= plt.plot(domain,range,color,lw=1,label=label)
		plt.legend(handler_map={p1: HandlerLine2D(numpoints=4)})

	def draw_points(self,array_points):
		plt.plot(array_points, "ob")

	def show_data(self):
		plt.show()

	def show_maximal(self,array_points):
		#snd_derivative = gaussian_filter(spectral_range,sigma=3.0,order=2)	
		#print [x for x in fst_derivative if x<0.01 and x>-0.01]}
		fst_derivative = gaussian_filter(array_points,sigma=5.0,order=1)
		fst_derivative_sorted = np.sort(fst_derivative)

		k = 10
		zeros = []
		for i in range(k):
			value = fst_derivative_sorted[i]
			index = np.where(fst_derivative==value)
			#index = fst_derivative.index(value)
			zeros.append(index[0][0])
		print zeros
		#zeros_values = [index for index,x in enumerate(zero) if x<0.001 and x>-0.001]
		#zeros_values = [index for index,x in enumerate(zero) ]
		#print zeros_values
		#wavelengths = [array_points[i] for i in zeros_values]
		return zeros
		#plt.plot(wavelength, zeros_values, 'ro')
		#plt.show()