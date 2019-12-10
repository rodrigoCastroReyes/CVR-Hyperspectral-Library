import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from HSCube import *
from skimage.measure import label

class HSLeafsDataframe(object):
	SIDE = 'lado'
	STAGE = 'estadio'
	LABELS = {
		"Sana" : 0,
		"Est1" : 1,
		"Est2" : 2,
		"Est3" : 3,
		"Est4" : 4,
		"Est5" : 5,
		"Est6" : 6,
		"Dano" : 7,
		"Inoc" : 10,
		"Otro" : 8,
		'Desconocido' : -1
	}

	def __init__(self,dir_input = None, type = 'file' , df = None ,stages = [0,1],include_cells=False):
		super(HSLeafsDataframe, self).__init__()
		if type == 'file':
			self.dir_input = dir_input
			self.df = pd.read_csv(self.dir_input)
			self.filter_by_stages(stages)
		elif type == 'cube':
			hype_cube = HSCube(dir_input=dir_input,type_cube = 'normalizado')
			self.df = self.transform_cube(hype_cube,include_cells)
		else:
			self.df = df

	def get_contours(self,hype_cube):
		result = hype_cube.get_mask()
		rows,cols = result.shape
		contour_img = np.zeros((rows,cols))
		label_img = label(result, connectivity=result.ndim)
		props = measure.regionprops(label_img)
		current_id = 1
		for prop in props:
			contour = prop["bbox"]
			area = prop["area"]
			if area > 50: 
				min_row, min_col, max_row, max_col = contour
				for row,col in itertools.product(range(min_row,max_row),range(min_col,max_col)):
					contour_img[row,col] = current_id
			current_id += 1

		return contour_img.flatten()

	def transform_cube(self,hype_cube,include_cells=False):
		self.rows = hype_cube.rows
		self.cols = hype_cube.cols
		mask = np.uint8(hype_cube.get_mask().flatten())
		sides = hype_cube.get_sides()
		sides = [ 'Derecho'  if side == 0 else 'Izquierdo' for side in sides]
		stages = [ 'Desconocido' for i in range(self.rows*self.cols)]
		semanas = [ 0 for i in range(self.rows*self.cols)]
		plana = [ 's' for i in range(self.rows*self.cols)]
		idRegions = [ -1 for i in range(self.rows*self.cols)]

		hype_data = hype_cube.data.reshape((self.rows*self.cols,520))
		positions = [ [row,col] for row,col in itertools.product(range(self.rows),range(self.cols))]
		
		data = np.column_stack((stages,sides,positions,semanas,plana,idRegions,mask,hype_data))
		features = ['estadio','lado','px','py','semana','plana','idRegion','no_background']
		for i in range(1,521):
			features.append( "l" + str(i))

		df = pd.DataFrame(data = data, columns = features)
		df['no_background'] = df['no_background'].astype(int)
		if include_cells:
			df['countour'] = self.get_contours(hype_cube)
		df = df[df['no_background']==1]
		#df = df.loc[df['no_background']==255]
		return df

	def get_positions(self):
		return self.df[['px','py']].iterrows()

	def filter_by_stages(self,ids):
		self.df = self.df[self.df[self.STAGE].isin(ids)]

	def solve_poly(self,wv,bin_spectral=2):
		c = 386.829 - wv
		b = 0.586*bin_spectral
		a = 0.000022*bin_spectral
		coeff = [a, b, c]
		solves = np.roots(coeff)
		solves = solves[solves>0]
		return int(math.floor(solves[0]))

	def get_side(self,side):
		df_side = self.df.loc[self.df[self.SIDE] == side]
		return HSLeafsDataframe(dir_input=None,df=df_side,type='dataframe')

	def get_spectral_range(self,wv_begin,wv_end):
		
		if (wv_begin >400) and (wv_begin < 1000) and (wv_end >400) and (wv_end < 1000):
			i_begin = self.solve_poly(wv_begin)
			j_end = self.solve_poly(wv_end)
			return self.df[self.df.columns[i_begin:j_end]]

		return None

	def get_SVIs(self):
		self.df['ir'] = self.df.apply(self.ir,axis=1)
		self.df['tcari'] = self.df.apply(self.tcari,axis=1)
		self.df['rgri'] = self.df.apply(self.rgri,axis=1)
		self.df['pri'] = self.df.apply(self.pri,axis=1)
		return self.df[['ir','tcari','rgri','pri']]

	def get_labels(self):
		self.df[self.STAGE] = self.df[self.STAGE].apply( lambda label: int(self.LABELS[label]))
		return self.df[self.STAGE]

	def get_wavelength(self,index,bin_spectral=2):
		return 0.000022*index*index*bin_spectral + 0.586*index*bin_spectral + 386.829

	def tcari(self,row):
		spectrum = row[7:527].values
		r_700 = np.mean(spectrum[248:290])#680,730
		r_670 = np.mean(spectrum[223:256])#650,690
		r_550 = np.mean(spectrum[130:148])#540,560
		if np.sum(r_670) == 0:
			return 0
		index =  3*((r_700 - r_670) - 0.2*(r_700-r_550))*(r_700/r_670)
		return index

	def ir(self,row):
		spectrum = row[7:527].values
		index = np.mean(spectrum[315:])
		return index

	def pri(self,row):
		spectrum = row[7:527].values
		r_570 = np.mean(spectrum[147:160])
		r_531 = np.mean(spectrum[117:139])#525,550
		index = 1.0*(r_531-r_570)/(r_531+r_570)
		#if (np.sum(r_531) == 0 ) and (np.sum(r_570) == 0) :
		#	return 0
		return index

	def rgri(self,row):
		spectrum = row[7:527].values
		#if ( np.sum(spectrum[97:181]) == 0 ):
		#	return 0
		index = np.sum(spectrum[181:265])/np.sum(spectrum[97:181])
		return index