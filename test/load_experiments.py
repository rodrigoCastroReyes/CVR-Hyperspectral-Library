import sys
sys.path.append("../")
from HSCube import *
from HSGeometricCalibration import *
import matplotlib.pyplot as plt


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