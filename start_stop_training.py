import matplotlib
from start_stop_training.callbacks.epochcheckpoint import EpochCheckpoint
from start_stop_training.callbacks.trainingmonitor import TrainingMonitor
from start_stop_training.nn.resnet import ResNet
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import cv2
import sys
import os

MODEL_PATH = None
CHECKPOINTS = 'start_stop_training_checkpoints'
START_EPOCH = None
BS = 128
EPOCHS = 80

# Load data, resize to expected Resnet input (32, 32), Normalize in range [0, 1], and add color channel
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX = np.array([cv2.resize(x, (32, 32)) for x in trainX])
testX = np.array([cv2.resize(x, (32, 32)) for x in testX])
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0
trainX = trainX.reshape((trainX.shape[0], 32, 32, 1))
testX = testX.reshape((testX.shape[0], 32, 32, 1))

# One-hot encode labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# Construct image data generator
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# If no existing model is supplied, compile model
if MODEL_PATH is None:
    print('[INFO] No model supplied: compiling new model')
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 1, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# If a model is supplied, load the model
else:
    print('[INFO] Model supplied: {}'.format(MODEL_PATH))
    model = load_model(MODEL_PATH)

    # Update learning rate
    print('[INFO] Existing learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print('[INFO] New learning rate: {}'.format(K.get_value(model.optimizer.lr)))

# Construct output paths
plotPath = 'resnet_fashion_mnist_plot.png'
jsonPath = 'resnet_fashion_mnist_json.json'

# Construct callbacks
callbacks = [EpochCheckpoint(CHECKPOINTS, every=5, startAt=START_EPOCH),
             TrainingMonitor(plotPath, jsonPath=jsonPath, startAt=START_EPOCH)]

# Train network
model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
          validation_data=(testX, testY),
          steps_per_epoch=len(trainX) // BS,
          epochs=EPOCHS,
          callbacks=callbacks,
          verbose=1)


