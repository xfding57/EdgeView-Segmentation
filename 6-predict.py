import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
import os, argparse
from imageio import get_writer
import util
from util import progressbar2 as progressbar
run = 1
present_dir = os.getcwd()


# arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-PATH', '--PATH', type=str, default='', help='input h5 dataset')
parser.add_argument('-SAVE', '--SAVE', type=str, default='', help='output folder')
parser.add_argument('-NN', '--NN', type=str, default='', help='model')
parser.add_argument('-outputbitrate', type=int, default=32, help='model')
parser.add_argument('-gpus', type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-depth', type=int, default=3, help='input depth (use for 3D CT image only)')
args, unparsed = parser.parse_known_args()


if len(unparsed) > 0:
	print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
	exit(0)


if len(args.gpus) > 0:
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR


# load neural network
NN = tf.keras.models.load_model(args.NN, )


# read noisy image to memory
ns_img_test = util.read_images(args.PATH)


# predict
idx = [s_idx for s_idx in range(ns_img_test.shape[0] - args.depth)]
dn_img = []
for s_idx in idx:
	X = np.array(np.transpose(ns_img_test[s_idx : (s_idx+args.depth)], (1, 2, 0)))
	X = X[np.newaxis, :, :, :]
	Y = NN.predict(X[:1])
	dn = Y[0,:,:,0]
	dn_img.append(dn)


# write denoised image stack to disk
sli = np.asarray(dn_img)
if not os.path.isdir(args.SAVE):
	os.mkdir(args.SAVE)
if args.outputbitrate == 16:
	sli = np.round(sli * 65535).astype(np.uint16)
for i in progressbar(range(np.shape(sli)[0]), "Saving: "):
	singlesli = sli[i,:,:]
	with get_writer(os.path.join(args.SAVE,"sli"+str(i).zfill(5)+".tif")) as writer:
		writer.append_data(singlesli, {"compress": 9})

