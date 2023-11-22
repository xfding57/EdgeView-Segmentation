import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from util import save2img, str2bool
import sys, os, time, argparse, shutil, scipy.io, glob
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from util import save2img, str2bool
from util import progressbar2 as progressbar
# import a generator model
from models import unet as make_generator_model
from data import bkgdGen, gen_train_batch_bg, get1batch4test
import tifffile
run = 1
present_dir = os.getcwd()


# arguments
parser = argparse.ArgumentParser(description='TomoGAN, for noise/artifact removal')
parser.add_argument('-PATH', '--PATH', type=str, required=True, help='read from')
parser.add_argument('-SAVE', '--SAVE', type=str, required=True, help='write to')
parser.add_argument('-bitrate', type=int, default=16, help='encoding of images')
parser.add_argument('-learningrate', type=float, default=1e-4, help='learning rate')
parser.add_argument('-gpus', type=str, default='0', help='')
parser.add_argument('-lmse', type=float, default=1, help='lambda mse')
parser.add_argument('-lunet', type=int, default=4, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input depth (use for 3D CT image only)')
parser.add_argument('-psz', type=int, default=64, help='cropping patch size')
parser.add_argument('-mbsz', type=int, default=32, help='mini-batch size')
parser.add_argument('-itg', type=int, default=1, help='iterations for G')
parser.add_argument('-epochs', type=int, default=1, help='number of epochs')
parser.add_argument('-epochsave', type=int, default=100, help='save result for every nth epoch')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
	print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
	exit(0)


# make log folder and get log number
if not os.path.isdir(args.SAVE):
	os.mkdir(args.SAVE)
LOG = os.path.join(args.SAVE,"_log")
if not os.path.isdir(LOG):
	os.mkdir(LOG)
lognum = len([f for f in os.listdir(LOG) if os.path.isfile(os.path.join(LOG,f))])
# record log
loglines = ["PATH = "+args.PATH \
	   ,"SAVE = "+args.SAVE \
	   ,"bitrate = "+str(args.bitrate) \
	   ,"learningrate = "+str(args.learningrate) \
	   ,"gpus = "+args.gpus \
	   ,"lmse = "+str(args.lmse) \
	   ,"lunet = "+str(args.lunet) \
	   ,"depth = "+str(args.depth) \
	   ,"psz = "+str(args.psz) \
	   ,"mbsz = "+str(args.mbsz) \
	   ,"itg = "+str(args.itg) \
	   ,"epochs = "+str(args.epochs)]
with open(os.path.join(LOG,"test"+str(lognum).zfill(4)+".txt"), 'w') as f:
	f.write('\n'.join(loglines))
# make output directory
# output will follow log number
SAVE = os.path.join(args.SAVE,"test"+str(lognum).zfill(4))
os.mkdir(SAVE)


# which GPU to use
if len(args.gpus) > 0:
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
# disable printing INFO, WARNING, and ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# build minibatch data generator with prefetch
mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(
	dsfn = args.PATH, \
	mb_size = args.mbsz, \
	in_depth = args.depth, \
	img_size = args.psz), \
	max_prefetch = args.mbsz*4)


generator = make_generator_model(input_shape=(None, None, args.depth), nlayers=args.lunet)
gen_optimizer = tf.keras.optimizers.Adam(args.learningrate)
LossG = np.zeros((args.epochs,1))


# inside the training loop
for epoch in progressbar(range(args.epochs), "Training: "):
	# Wrap the training loop in a try block
	try:
		time_git_st = time.time()
		for _ge in range(args.itg):
			X_mb, y_mb = mb_data_iter.next() # with prefetch
			with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
				gen_tape.watch(generator.trainable_variables)
				gen_imgs = generator(X_mb, training=True)

				gen_loss = 0

				# use bce as loss function
				if run == 0:
					if args.bitrate == 8:
						X_mb_norm = X_mb / 255
						y_mb_norm = y_mb / 255
					if args.bitrate == 16:
						X_mb_norm = X_mb / 65535
						y_mb_norm = y_mb / 65535
					# loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb_norm)
					loss_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(gen_imgs, y_mb_norm)
					gen_loss = args.lmse * loss_bce
					# gen_loss = gen_loss + (loss_bce * loss_weights[checklossfunc])

				# use ncc as loss function
				if run == 0:
					# print(">>> Convert images to tensors")
					image1 = tf.convert_to_tensor(gen_imgs, dtype=tf.float32)
					image2 = tf.convert_to_tensor(y_mb, dtype=tf.float32)
					# Normalize pixel values to the range [0, 1]
					if args.bitrate == 16:
						image1 = image1 / 65535.0  # Assuming 16-bit images (0-65535)
						image2 = image2 / 65535.0  # Normalize to the range [0, 1]
					# print(">>> Calculate means of each image")
					mean1 = tf.reduce_mean(image1)
					mean2 = tf.reduce_mean(image2)
					# print(">>> Calculate cross-correlation")
					cross_corr = tf.reduce_sum((image1 - mean1) * (image2 - mean2))
					# print(">>> Calculate standard deviations")
					std1 = tf.sqrt(tf.reduce_sum(tf.square(image1 - mean1)))
					std2 = tf.sqrt(tf.reduce_sum(tf.square(image2 - mean2)))
					loss_ncc = - (cross_corr / (std1 * std2))
					gen_loss = args.lmse * loss_ncc
					# gen_loss = gen_loss + (loss_ncc * loss_weights[checklossfunc])

				# use mse as loss function
				if run == 0:
					if args.bitrate == 8:
						X_mb_norm = X_mb / 255
						y_mb_norm = y_mb / 255
					if args.bitrate == 16:
						X_mb_norm = X_mb / 65535
						y_mb_norm = y_mb / 65535
					# loss_mse = tf.keras.losses.MeanSquaredError()(gen_imgs, y_mb_norm)
					loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb_norm)
					gen_loss = args.lmse * loss_mse
					# gen_loss = gen_loss + (loss_mse * loss_weights[checklossfunc])

				# use ssim for loss function
				if run == 1:
					loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb)
					# loss_l1 = tf.losses.mean_absolute_error(gen_imgs, y_mb)
					if args.bitrate == 8:
						loss_ssim = (tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=255))
					elif args.bitrate == 16:
						loss_ssim = (tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=65535))
					else:
						loss_ssim = (tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=255))
					# gen_loss = args.lmse * (1-loss_ssim)
					gen_loss = args.lmse * (1-loss_ssim)


			gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
			gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

		# Save loss
		LossG[epoch] = gen_loss.numpy().mean()
		# save as txt
		with open(os.path.join(SAVE,"LossG.txt"), 'a') as file:
			file.write(str(float(LossG[epoch])) + '\n')
		# save as mat
		scipy.io.savemat(os.path.join(SAVE,"LossG.mat"), {'LossG':np.array(LossG)})

		X222, y222 = get1batch4test(dsfn=args.PATH, in_depth=args.depth)
		if epoch == 0:
			# save ground truth and target
			tifffile.imwrite(os.path.join(SAVE,"it_target.tif"), y222[0, :, :, 0])
			tifffile.imwrite(os.path.join(SAVE,"it_input.tif"), X222[0, :, :, args.depth // 2])
		if epoch == 0 or epoch == (args.epochs-1) or epoch%args.epochsave == 0:
			pred_img = generator.predict(X222[:1])
			# save image
			tifffile.imwrite(os.path.join(SAVE,"it"+str(epoch).zfill(5)+".tif"), pred_img[0, :, :, 0])
			# save model
			generator.save(os.path.join(SAVE,"it"+str(epoch).zfill(5)+".h5"), include_optimizer=False)

		sys.stdout.flush()

	except KeyboardInterrupt:
		print("Training interrupted by the user. Generating the plot...")
		break  # Exit the training loop


# Load the LossG values from the text file
loss_data = np.loadtxt(os.path.join(SAVE,"LossG.txt"))
# Plotting the LossG values
plt.figure(figsize=(12, 8))
# plt.plot(range(args.epochs), loss_data, linewidth=2.5)
plt.plot(range(np.size(loss_data)), loss_data, linewidth=2.5)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Save the plot as a PNG file
plt.savefig(os.path.join(SAVE,"loss_plot.png"), dpi=300, bbox_inches='tight')
# Clear the current figure
plt.clf()
