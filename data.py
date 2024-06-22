import numpy as np
import queue as Queue
import threading
import util
import os
import tifffile


class bkgdGen(threading.Thread):
	def __init__(self, data_generator, max_prefetch=1):
		threading.Thread.__init__(self)
		self.queue = Queue.Queue(max_prefetch)
		self.generator = data_generator
		self.daemon = True
		self.start()
	def run(self):
		for item in self.generator:
			# block if necessary until a free slot is available
			self.queue.put(item, block=True, timeout=None)
		self.queue.put(None)
	def next(self):
		# block if necessary until an item is available
		next_item = self.queue.get(block=True, timeout=None)
		if next_item is None:
			raise StopIteration
		return next_item
	# Python 3 compatibility
	def __next__(self):
		return self.next()
	def __iter__(self):
		return self


def gen_train_batch_bg(PATH, mb_size, in_depth, img_size):
	# read from folder
	X = util.read_images(os.path.join(PATH,"B")).astype(np.float32)
	Y = util.read_images(os.path.join(PATH,"A")).astype(np.float32)
	while True:
		idx = np.random.randint(0, X.shape[0]-in_depth, mb_size)
		if img_size == X.shape[1]:
			# preallocate splace with zero array
			rst, cst = np.zeros(mb_size, dtype=np.int), np.zeros(mb_size, dtype=np.int)
		else:
			rst = np.random.randint(0, X.shape[1]-img_size, mb_size)
			cst = np.random.randint(0, X.shape[2]-img_size, mb_size)
		batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
		batch_X = [batch_X[_i, _r:_r+img_size, _c:_c+img_size, :] for _i, _r, _c in zip(range(mb_size), rst, cst)]
		batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3)
		batch_Y = [batch_Y[_i, _r:_r+img_size, _c:_c+img_size, :] for _i, _r, _c in zip(range(mb_size), rst, cst)]
		yield np.array(batch_X), np.array(batch_Y)


def get1batch4test(PATH, in_depth):
	# read from folder
	X = util.read_images(os.path.join(PATH,"B-test")).astype(np.float32)
	Y = util.read_images(os.path.join(PATH,"A-test")).astype(np.float32)
	# always use slice in_depth//2 for validation
	idx = (X.shape[0]-in_depth, )
	batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
	batch_Y = np.expand_dims([Y[s_idx+in_depth//2] for s_idx in idx], 3)
	return batch_X.astype(np.float32) , batch_Y.astype(np.float32)


def save_model_and_images(epoch, generator, X222, SAVE, args):
    """ Save the model and images at specified intervals """
    if epoch == 0 or epoch == (args.epochs - 1) or epoch % args.epochsave == 0:
        pred_img = generator.predict(X222[:1])
        # Save image
        tifffile.imwrite(os.path.join(SAVE, "it" + str(epoch).zfill(5) + ".tif"), pred_img[0, :, :, 0])
        # Save model
        generator.save(os.path.join(SAVE, "it" + str(epoch).zfill(5) + ".h5"), include_optimizer=False)

