import util
import os
import numpy as np
from skimage.filters import threshold_otsu
import tifffile

# arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-PATH', '--PATH', type=str, required=True, help='Directory to read from')
parser.add_argument('-SAVE', '--SAVE', type=str, required=True, help='Directory to write to')
parser.add_argument('-histperc', type=float, default=0.995, help='')
parser.add_argument('-expansion', type=float, default=0.33, help='')
parser.add_argument('-mode', type=int, default=1, help='')
parser.add_argument('-teststackstart', type=int, default=1, help='')
parser.add_argument('-teststackthickness', type=int, default=1, help='')
args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
	print("Unrecognized argument")
	exit(0)

PATH = args.PATH
SAVE = args.SAVE
histperc = args.histperc
expansion = args.expansion
mode = args.mode

im = util.read_images(PATH)
# Flatten the image stack to get the pixel intensity values across all images
pixels = im.flatten()
# lowest and highest values
lowest = np.min(im)
highest = np.max(im)

# Assuming pixels is your flattened array from the TIFF stack
low = np.percentile(pixels, (1-histperc)*100)
high = np.percentile(pixels, histperc*100)
range_ = high - low
# Calculate expansion of the range
expansion = expansion * range_
# Expand the low boundary downwards and the high boundary upwards by the expansion value
lower = low - expansion
higher = high + expansion

# scale im
if mode == 1:
	im_partialhist = ((im-lowest)/(highest-lowest)*65535).astype(np.uint16)
elif mode == 2:
	im_partialhist = ((im-lower)/(higher-lower)*65535).astype(np.uint16)
elif mode == 3:
	im_partialhist = ((im-low)/(high-low)*65535).astype(np.uint16)

# make output folder if it doesn't exist
if not os.path.isdir(SAVE):
	os.mkdir(SAVE)

# Iterate over each image in the scaled stack
for i, im_single in enumerate(im_partialhist):
    filename = os.path.join(SAVE,"im_"+str(i).zfill(4)+".tif")
    tifffile.imwrite(filename, im_single)

# make output test folder if it doesn't exist
if not os.path.isdir(SAVE+"-test"):
	os.mkdir(SAVE+"-test")

# Iterate over each image in the scaled stack
for i, im_single in np.arange(teststackstart,teststackthickness+1,1):
    filename = os.path.join(SAVE+"-test","im_"+str(i).zfill(4)+".tif")
    tifffile.imwrite(filename, im_single)

