from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile


class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        # Store model, stop factor, and beta to create a smoothed average loss curve
        self.model = model
        self.stopFactor = stopFactor  # When to terminate training because loss has increased too much
        self.beta = beta

        # Initialize learning rate and loss lists
        self.lrs = []
        self.losses = []

        # Initialize learning rate multiplier, average loss, best loss found, current batch, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # Reset all values
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_generator(self, data):
        # Define image data input types
        iterClasses = ['NumpyArrayIterator',
                       'DictionaryIterator',
                       'DataframeIterator',
                       'Iterator',
                       'Sequence']

        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        # Extract current learning rate and append list
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Extract current loss, increment batch count, compute and smooth average loss, and append list
        l = logs['loss']
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)

        # Compute max loss stopping point
        stopLoss = self.stopFactor * self.bestLoss

        # Check if loss surpasses stop threshold
        if self.batchNum > 1 and smooth > stopLoss:
            # Stop training
            self.model.stop_training = True
            return

        # Check if best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth

        # Increase learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, trainData, startLR, endLR, epochs=None, stepsPerEpoch=None,
             batchSize=32, sampleSize=2048, verbose=1):
        # Reset variables
        self.reset()

        # Determine if data generator is used
        useGen = self.is_data_generator(trainData)

        # If data generator is used and steps per epoch is not supplied raise error
        if useGen and stepsPerEpoch is None:
            msg = 'stepsPerEpoch=None: When using data generator, steps per epoch must be supplied'
            raise Exception(msg)

        elif not useGen:
            # Extract number of samples in train data and compute steps per epoch
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))

        # If number of epochs is not supplied, compute it
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

        # Compute number of batch updates
        numBatchUpdates = epochs * stepsPerEpoch

        # Compute learning rate multiplier
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)

        # Construct temporary file path for weights
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)

        # Extract original learning rate and set starting learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)

        # Construct callback at batch end
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        # Check for data generator
        if useGen:
            self.model.fit(x=trainData,
                           steps_per_epoch=stepsPerEpoch,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=[callback])

        else:
            self.model.fit(x=trainData[0],
                           y=trainData[1],
                           batchSize=batchSize,
                           epochs=epochs,
                           callbacks=[callback],
                           verbose=verbose)

        # Restore original model weights and learning rate after training
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=''):
        # Extract learning rates and losses
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]

        # Plot learning rates and losses
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        if title != '':
            plt.title(title)
