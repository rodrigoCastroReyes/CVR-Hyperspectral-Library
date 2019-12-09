import sys
sys.path.append("/home/rodfcast/Documents/CVR/Proyecto RIP/Hyperspectral_Library")
from HSImageWorker import *
from HSCube import *

def test_hsimageworker(dir_input):
	iw = HSImageWorker(dir_input,spectral_range='nir')
	images = iw.load()
	print(images.shape)

dir_input = sys.argv[1]
cube = HSCube(dir_input)
print(cube.data.shape)