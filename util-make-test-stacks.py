import util
import os
import numpy as np
from skimage.filters import threshold_otsu
import tifffile

def get_grey_boundary(PATH,histperc,expansion):

	im = util.read_images(PATH)
	# Flatten the image stack to get the pixel intensity values across all images
	pixels = im.flatten()
	# lowest and highest values
	lowest = np.min(im)
	highest = np.max(im)

	# print(lowest)
	# print(highest)

	# Assuming pixels is your flattened array from the TIFF stack
	low = np.percentile(pixels, (1-histperc)*100)
	high = np.percentile(pixels, histperc*100)
	range_ = high - low
	# Calculate expansion of the range
	expansion = expansion * range_
	# Expand the low boundary downwards and the high boundary upwards by the expansion value
	lower = low - expansion
	higher = high + expansion

	print([lowest,lower,low,high,higher,highest])

def set_grey_boundary(PATH,SAVE,histperc,expansion,mode):

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


PATH = "/staff/dingx/Desktop/Local_data/dingx/prj35G12338-Hydrogels/rec/Xiaofan/2-EdgeView/240215-samples-sc67-sc88-segment-for-motionmodel/sc67-edgeview/2/test/0001/sli"
SAVE = "/staff/dingx/Desktop/Local_data/dingx/prj35G12338-Hydrogels/rec/Xiaofan/2-EdgeView/240215-samples-sc67-sc88-segment-for-motionmodel/sc67-edgeview/2/test/A-1"
histperc = 0.995 # 99% of histogram
expansion = 0.33 # expand 30% of histogram range

set_grey_boundary(PATH,histperc,expansion,1)




