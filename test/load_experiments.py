import sys
sys.path.append("../")
from HSCube import *
from HSGeometricCalibration import *
import matplotlib.pyplot as plt


<<<<<<< HEAD
geo_normalization = Polder() #choose geometric normalization that you want to apply
#geo_normalization = SNV()

#pass a 'HSGeometricCalibration object' to HSCube in order to normalize each pixel in cube
cube = HSCube(dir_input,type_cube='normalizado')#,geometric_calibrator=geo_normalization)
#call do_geometric_calibration to perform operation
#cube.do_geometric_calibration()

#you can select a specific pixel to draw
spectrum = cube.get_spectrum_by_pixel(667,448)
wavelengths = cube.get_wavelengths()
plt.plot(wavelengths,spectrum)
plt.show()
=======
dir_input = sys.argv[1]#input a directory of experiment
px = int(sys.argv[2])
py = int(sys.argv[3])
min_wv = int(sys.argv[4])#en nm
max_wv = int(sys.argv[5])#en nm

cube = HSCube(dir_input,type_cube='normalized',min_wv=min_wv,max_wv=max_wv)
if cube.load_data() :
    #you can select a specific pixel to draw
    spectrum = cube.get_spectrum_by_pixel(px,py)
    wavelengths = cube.get_wavelengths(bin_spectral=1)
    plt.plot(wavelengths,spectrum)
    plt.show()
else:
    print("No data :(")
>>>>>>> 3e3e26721b52d9725d7524ac37d3f6709f89367c
