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
mask = cube.get_mask()

fig, ax = plt.subplots(1,4,figsize=(10, 6))
ax[0].imshow(cube.get_ir(),interpolation="nearest")
ax[0].set_axis_off()
ax[1].imshow(cube.leaf,interpolation="nearest")
ax[1].set_axis_off()
ax[2].imshow(cube.background,interpolation="nearest")
ax[2].set_axis_off()
ax[3].imshow(cube.mask,interpolation="nearest")
ax[3].set_axis_off()
plt.tight_layout()
plt.show()