import sys
sys.path.append("/home/rodfcast/Documents/CVR/Proyecto RIP/Hyperspectral_Library")
from HSCube import *
from HSGeometricCalibration import *
import matplotlib.pyplot as plt

dir_input = sys.argv[1]#input a directory of experiment


geo_normalization = Polder() #choose geometric normalization that you want to apply
#geo_normalization = SNV()

#pass a 'HSGeometricCalibration object' to HSCube in order to normalize each pixel in cube
cube = HSCube(dir_input,type_cube='raw',geometric_calibrator=geo_normalization)
#call do_geometric_calibration to perform operation
cube.do_geometric_calibration()

#you can select a specific pixel to draw
spectrum = cube.get_spectrum_by_pixel(100,50)
wavelengths = cube.get_wavelengths()
plt.plot(wavelengths,spectrum)
plt.show()