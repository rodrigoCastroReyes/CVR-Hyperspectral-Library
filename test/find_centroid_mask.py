import sys
sys.path.append("/home/rodfcast/Documents/CVR/Proyecto RIP/Hyperspectral_Library")
from HSImageWorker import *
from HSCube import *
from skimage import measure
import matplotlib.pyplot as plt
import tifffile as tiff
import matplotlib.patches as patches
from skimage import draw

dir_input = sys.argv[1]
cube = HSCube(dir_input,type_cube='raw')
mask = cube.get_background(binary_not=False)
label_image = measure.label(mask)

fig, ax = plt.subplots(figsize=(10, 6))

for region in measure.regionprops(label_image):
	minr, minc, maxr, maxc = region.bbox
	(py,px) = region.centroid
	rr, cc = draw.circle(py, px, 5)
	mask[rr,cc] = 0
	rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
	ax.add_patch(rect)
	ax.set_axis_off()
	plt.tight_layout()
ax.imshow(mask)
plt.show()