import sys
sys.path.append("../")
from HSCube import *
from HSGeometricCalibration import *
import matplotlib.pyplot as plt

#to run: python load_experiments.py path_to_folder_experiment
#example Experimento-2017-03-08\ 18-39-32/

dir_input = sys.argv[1]#input a directory of experiment

geo_normalization = Polder()#choose a geometric normalization method
#geo_normalization = SNV()

#pass a 'HSGeometricCalibration object' to HSCube in order to normalize each pixel in cube
cube = HSCube(dir_input,type_cube='raw',geometric_calibrator=geo_normalization)
#call do_geometric_calibration to perform operation
cube.do_geometric_calibration()

#select a specific pixel to draw
spectrum = cube.get_spectrum_by_pixel(100,50)
wavelengths = cube.get_wavelengths()
plt.plot(wavelengths,spectrum)
plt.show()