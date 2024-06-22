import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sys, os, time, argparse, scipy.io, glob
import matplotlib.pyplot as plt
from util import save2img, str2bool, parselist, plot_loss
from util import progressbar2 as progressbar
# import a generator model
from models import unet as make_generator_model
import data
import tifffile
run = 1

# arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('-PATH', '--PATH', type=str, required=True, help='Directory to read from')
parser.add_argument('-SAVE', '--SAVE', type=str, required=True, help='Directory to write to')
parser.add_argument('-bitrate', type=int, default=16, help='Bitrate encoding of inputs')
parser.add_argument('-gpus', type=str, default='0', help='Which GPU to use')
parser.add_argument('-weights', type=parselist, default='0,0,0,1', help='List of initial weights for loss functions in format 1,2,3')
parser.add_argument('-learningrate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('-lmse', type=float, default=1, help='lambda mse')
parser.add_argument('-lunet', type=int, default=4, help='Number of UNet layers')
parser.add_argument('-depth', type=int, default=3, help='Input depth (use for 3D CT image only)')
parser.add_argument('-patchsize', type=int, default=64, help='Cropping patch size')
parser.add_argument('-batchsize', type=int, default=32, help='Mini-batch size')
parser.add_argument('-itg', type=int, default=1, help='Number of iterations for training the generator in each epoch')
parser.add_argument('-epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('-epochsave', type=int, default=100, help='Save result for every nth epoch')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
	print("Unrecognized argument")
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
	   ,"gpus = "+args.gpus \
	   ,"training weights = "+str(args.weights[0])+" l1 "+str(args.weights[1])+" mse "+str(args.weights[2])+" ncc "+str(args.weights[3])+" ssim" \
	   ,"learning rate = "+str(args.learningrate) \
	   ,"lunet = "+str(args.lunet) \
	   ,"depth = "+str(args.depth) \
	   ,"patch size = "+str(args.patchsize) \
	   ,"mini batch size = "+str(args.batchsize) \
	   ,"itg = "+str(args.itg) \
	   ,"epochs = "+str(args.epochs) \
	   ,"epochs save = "+str(args.epochsave)]
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
mb_data_iter = data.bkgdGen(data_generator = data.gen_train_batch_bg(
	PATH = args.PATH, \
	mb_size = args.batchsize, \
	in_depth = args.depth, \
	img_size = args.patchsize), \
	max_prefetch = args.batchsize*4)

generator = make_generator_model(input_shape=(None, None, args.depth), num_layers=args.lunet)
gen_optimizer = tf.keras.optimizers.Adam(args.learningrate)
Lossmse = np.zeros((args.epochs,1))
Lossl1 = np.zeros((args.epochs,1))
Lossncc = np.zeros((args.epochs,1))
Lossssim = np.zeros((args.epochs,1))
LossG = np.zeros((args.epochs,1))

if run == 1:
	# inside the training loop
	for epoch in progressbar(range(args.epochs), "Training: "):

		# warap the training loop in a try block
		try:
			time_git_st = time.time()
			for _ge in range(args.itg):

				# with prefetch
				X_mb, y_mb = mb_data_iter.next()

				with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
					gen_tape.watch(generator.trainable_variables)
					gen_imgs = generator(X_mb, training=True)
					# gen_loss = 0
					# gen_loss = tf.constant(0, dtype=tf.float32)

					if args.bitrate == 8:
						X_mb_norm = X_mb / 255
						y_mb_norm = y_mb / 255
					elif args.bitrate == 16:
						X_mb_norm = X_mb / 65535
						y_mb_norm = y_mb / 65535
					elif args.bitrate == 32:
						X_mb_min, X_mb_max = tf.reduce_min(X_mb),tf.reduce_max(X_mb)
						y_mb_min, y_mb_max = tf.reduce_min(y_mb),tf.reduce_max(y_mb)
						X_mb_norm = (X_mb-X_mb_min)/(X_mb_max-X_mb_min)
						y_mb_norm = (y_mb-y_mb_min)/(y_mb_max-y_mb_min)

					# calculate msa (l1) loss
					loss_l1 = args.weights[0]*(tf.keras.losses.mean_absolute_error (gen_imgs, y_mb))
					loss_l1_scalar = tf.reduce_mean(loss_l1) # reduce to scaler

					# calculate mse (l2) loss
					loss_mse = args.weights[1]*(tf.losses.mean_squared_error(gen_imgs, y_mb_norm))
					loss_mse_scalar = tf.reduce_mean(loss_mse) # reduce to scaler

					# calculate bce loss
					# loss_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(gen_imgs, y_mb_norm)

					# calculate ncc loss
					# convert images to tensors
					image1 = tf.convert_to_tensor(gen_imgs, dtype=tf.float32)
					image2 = tf.convert_to_tensor(y_mb, dtype=tf.float32)
					# Normalize pixel values to the range [0, 1]
					if args.bitrate == 8:
						image1 = image1 / 255
						image2 = image2 / 255
					if args.bitrate == 16:
						image1 = image1 / 65535
						image2 = image2 / 65535
					if args.bitrate == 32:
						image1_min, image1_max = tf.reduce_min(image1),tf.reduce_max(image1)
						image2_min, image2_max = tf.reduce_min(image2),tf.reduce_max(image2)
						image1 = (image1-image1_min)/(image1_max-image1_min)
						image2 = (image2-image2_min)/(image2_max-image2_min)
					# calculate means of each image
					mean1 = tf.reduce_mean(image1)
					mean2 = tf.reduce_mean(image2)
					# calculate cross-correlation
					cross_corr = tf.reduce_sum((image1 - mean1) * (image2 - mean2))
					# calculate standard deviations
					std1 = tf.sqrt(tf.reduce_sum(tf.square(image1 - mean1)))
					std2 = tf.sqrt(tf.reduce_sum(tf.square(image2 - mean2)))
					loss_ncc = args.weights[2]*(-(cross_corr / (std1 * std2)))

					# calculate ssim loss
					if args.bitrate == 8:
						loss_ssim = (tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=255))
					elif args.bitrate == 16:
						loss_ssim = (tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=65535))
					elif args.bitrate == 32:
						loss_ssim = (tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=1))
					loss_ssim = args.weights[3]*(1-loss_ssim)
					loss_ssim_scalar = tf.reduce_mean(loss_ssim) # reduce to scaler

					# calculate combined losses
					gen_loss = loss_l1_scalar + loss_mse_scalar + loss_ncc + loss_ssim_scalar

				# apply gradients
				gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
				gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

			# save loss as txt
			Lossmse[epoch] = loss_mse.numpy().mean()
			with open(os.path.join(SAVE,"Lossmse.txt"), 'a') as file:
				file.write(str(float(Lossmse[epoch])) + '\n')
			Lossl1[epoch] = loss_l1.numpy().mean()
			with open(os.path.join(SAVE,"Lossl1.txt"), 'a') as file:
				file.write(str(float(Lossl1[epoch])) + '\n')
			Lossncc[epoch] = loss_ncc.numpy().mean()
			with open(os.path.join(SAVE,"Lossncc.txt"), 'a') as file:
				file.write(str(float(Lossncc[epoch])) + '\n')
			Lossssim[epoch] = loss_ssim.numpy().mean()
			with open(os.path.join(SAVE,"Lossssim.txt"), 'a') as file:
				file.write(str(float(Lossssim[epoch])) + '\n')
			LossG[epoch] = gen_loss.numpy().mean()
			with open(os.path.join(SAVE,"LossG.txt"), 'a') as file:
				file.write(str(float(LossG[epoch])) + '\n')

			X222, y222 = data.get1batch4test(PATH=args.PATH, in_depth=args.depth)
			# save input and target
			if epoch == 0:
				tifffile.imwrite(os.path.join(SAVE,"it_target.tif"), y222[0, :, :, 0])
				tifffile.imwrite(os.path.join(SAVE,"it_input.tif"), X222[0, :, :, args.depth // 2])

			# save model according to -epochsave
			data.save_model_and_images(epoch, generator, X222, SAVE, args)

		except KeyboardInterrupt:
			print("Training interrupted by the user. Generating the plot...")
			break  # Exit the training loop

	# plot loss
	plot_loss(os.path.join(SAVE, "Lossmse.txt"), SAVE)
	plot_loss(os.path.join(SAVE, "Lossl1.txt"), SAVE)
	plot_loss(os.path.join(SAVE, "Lossncc.txt"), SAVE)
	plot_loss(os.path.join(SAVE, "Lossssim.txt"), SAVE)
	plot_loss(os.path.join(SAVE, "LossG.txt"), SAVE)


