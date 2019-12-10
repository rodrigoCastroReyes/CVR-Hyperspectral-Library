import sys
sys.path.append("/home/rodfcast/Documents/CVR/Proyecto RIP/Hyperspectral_Library")
from HSCube import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import draw

dir_input = sys.argv[1]
df = pd.read_csv(dir_input)
for index in range(len(df)):
	dir_input = df.ix[index]['dir_input']
	px = df.ix[index]['px']
	py = df.ix[index]['py']
	rx = df.ix[index]['rx']
	ry = df.ix[index]['ry']
	label = df.ix[index]['label']
	cube = HSCube(dir_input,type_cube='raw',spectral_range_flag='all')
	side = cube.get_side_of_leaf(px,py,rx,ry)
	print ("Label: %s, Side: %s"%(label,side))