import sys
sys.path.append("/home/rodfcast/Documents/CVR/Proyecto RIP/Hyperspectral_Library")
from HSGeometricCalibration import *

dir_input = sys.argv[1]

#directories = [ os.path.join(dir_input,f) for f in os.listdir(dir_input) if os.path.isdir(os.path.join(dir_input, f))]

#for dir in directories:
hs_geo = Polder(dir_input)
hs_geo.calibrate()
hs_geo.save()