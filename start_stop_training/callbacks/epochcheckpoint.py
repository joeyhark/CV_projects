from tensorflow.keras.callbacks import Callback
import os


class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		# Call the parent constructor
		super(Callback, self).__init__()

		# Store model output path, epochs to run before saving, and current epoch
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# Check if model should be saved
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath, "epoch_{}.hdf5".format(self.intEpoch + 1)])
			self.model.save(p, overwrite=True)

		# Increment epoch counter
		self.intEpoch += 1
