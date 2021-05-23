from LRF_and_Architectures.learning_rate_finder import LearningRateFinder
from LRF_and_Architectures.minigooglenet import MiniGoogLeNet
from LRF_and_Architectures.clr_callback import CyclicLR
from LRF_and_Architectures import config_MNIST
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

# Load data, resize to (32, 32), normalize to range [0, 1]
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX = np.array([cv2.resize(x, (32, 32)) for x in trainX])
testX = np.array([cv2.resize(x, (32, 32)) for x in testX])
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0
trainX = trainX.reshape((trainX.shape[0], 32, 32, 1))
testX = testX.reshape((testX.shape[0], 32, 32, 1))

# One-hot encode labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Initialize image data generator
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# Compile model
opt = SGD(lr=config_MNIST.MIN_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Check if attempt to find optimal learning rate
if config_MNIST.LR_FIND:
    print('[INFO] finding learning rate')
    # Initialize learning rate finder in range [1e-10, 1e+1]
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=config_MNIST.BATCH_SIZE), 1e-10, 1e+1,
             stepsPerEpoch=np.ceil((len(trainX) / float(config_MNIST.BATCH_SIZE))),
             batchSize=config_MNIST.BATCH_SIZE)

    # Plot loss against learning rate
    lrf.plot_loss()
    plt.savefig(config_MNIST.LRFIND_PLOT_PATH)

    # Exit script
    sys.exit(0)

# If learning rate range is known, initialize cyclic learning rate method
stepSize = config_MNIST.STEP_SIZE * (trainX.shape[0] // config_MNIST.BATCH_SIZE)
clr = CyclicLR(mode=config_MNIST.CLR_METHOD, base_lr=config_MNIST.MIN_LR, max_lr=config_MNIST.MAX_LR, step_size=stepSize)

# Train model
H = model.fit(x=aug.flow(trainX, trainY, batch_size=config_MNIST.BATCH_SIZE),
              validation_data=(testX, testY),
              steps_per_epoch=trainX.shape[0] // config_MNIST.BATCH_SIZE,
              epochs=config_MNIST.NUM_EPOCHS,
              callbacks=[clr],
              verbose=1)

# Evaluate model and print classification report
preds = model.predict(x=testX, batch_size=config_MNIST.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=config_MNIST.CLASSES))

# Construct training plot
N = np.arange(0, config_MNIST.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config_MNIST.TRAINING_PLOT_PATH)

# Construct learning rate plot
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config_MNIST.CLR_PLOT_PATH)
