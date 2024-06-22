import imageio, os, sys, PIL, re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import signal
import logging


def root_path():
	return os.path.abspath(os.sep)


def progressbar2(it, prefix="", size=50, out=sys.stdout):
	count = len(it)
	def show(j):
		x = int(size*j/count)
		out.write("%s[%s%s] %i/%i\r" % (prefix, u"#"*x, "."*(size-x), j, count))
		out.flush()        
	show(0)
	for i, item in enumerate(it):
		yield item
		show(i+1)
	out.write("\n")
	out.flush()


def read_images(image_directory):
	""" This function takes the directory containing darks, flats, and tomo folder then returns 3D arrays for darks, flats, and tomo """
	# get names of all files inside directory
	directory_file_names = sorted_nicely([f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))])
	# open the first image
	image_file = PIL.Image.open(os.path.join(image_directory, directory_file_names[0]))
	# change from image to array
	image_as_array = np.array(image_file)
	# get the shape of the array
	image_as_array_shape = np.shape(image_as_array)
	# preallocate space
	preallocate_matrix = np.zeros((len(directory_file_names), image_as_array_shape[0], image_as_array_shape[1]))
	# open all images and save as 3d array
	for i in range(len(directory_file_names)):
		image_file = PIL.Image.open(os.path.join(image_directory, directory_file_names[i]))
		preallocate_matrix[i] = np.array(image_file)
	image_stack_array = preallocate_matrix
	return image_stack_array


def delete_position(read_stack, slice_thickness, slice_position):
	""" This function reduces the 3d array stack of images, to just the specified slices for 
	which reconstruction will be performed """
	# Get dimensions of projections
	num_projections, num_height, num_width = np.shape(read_stack)
	vector = np.arange(slice_thickness)+slice_position
	delete_vector = np.arange(num_height)
	delete_vector = np.delete(delete_vector, vector)
	# Preallocate
	preallocate_matrix = np.zeros((num_projections, slice_thickness, num_width))
	for i in range(num_projections):
		stack = read_stack[i,:,:]
		stack = np.delete(stack, delete_vector, 0)
		preallocate_matrix[i,:,:] = stack
	return preallocate_matrix


def sorted_nicely( l ):
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)


def save2img(d_img, fn):
    # _min, _max = d_img.min(), d_img.max()
    # if np.abs(_max - _min) < 1e-4:
    #     img = np.zeros(d_img.shape)
    # else:
        # img = (d_img - _min) * 255. / (_max - _min)
    img = d_img
    img = img.astype('uint8')
    imageio.imwrite(fn, img)


def scale2uint8(d_img):
	# min, _max = d_img.min(), d_img.max()
    np.nan_to_num(d_img, copy=False)
    _min, _max = np.percentile(d_img, 0.05), np.percentile(d_img, 99.95)
    s_img = d_img.clip(_min, _max)
    if _max == _min:
        s_img -= _max
    else:
        s_img = (s_img - _min) * 255. / (_max - _min)
    return s_img.astype('uint8')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parselist(string):
	'''Converts a comma-separated string into a list of floats or integers.'''
	try:
		return [float(num) if '.' in num else int(num) for num in string.split(',')]
	except ValueError:
		raise argparse.ArgumentTypeError("Invalid format for --weights, expected format: 1,2,3")



def plot_loss(file_path, save_dir):
	""" Plot the loss from a text file and save the plot as an image """
	# Check if the loss file exists
	if os.path.exists(file_path):
		# Load the loss data
		loss_data = np.loadtxt(file_path)
		# Plotting the Loss
		plt.figure(figsize=(12, 8))
		plt.plot(range(len(loss_data)), loss_data, linewidth=2.5)
		plt.xlabel('Epochs', fontsize=20)
		plt.ylabel('Loss', fontsize=20)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		# Save the plot as a PNG file
		plot_path = os.path.join(save_dir, os.path.splitext(file_path)[0]+"_plot.png")
		plt.savefig(plot_path, dpi=300, bbox_inches='tight')
		# Clear the current figure
		plt.clf()
	else:
		print("Loss file not found.")
		return

