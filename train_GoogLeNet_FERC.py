from LRF_and_Architectures.learning_rate_finder import LearningRateFinder
from LRF_and_Architectures.minigooglenet import MiniGoogLeNet
from LRF_and_Architectures.clr_callback import CyclicLR
from LRF_and_Architectures import config_FERC
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from imutils import paths
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os

# Initialize preprocessed data and label lists
data = []
labels = []

# Extract and shuffle image paths
imagePaths = sorted(list(paths.list_images(config_FERC.INPUT_DATASET)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # Load image, resize, convert to array, and append list
    image = cv2.imread(imagePath)
    resized = cv2.resize(image, (config_FERC.IMAGE_DIMS[1], config_FERC.IMAGE_DIMS[0]))
    img_array = img_to_array(resized)
    data.append(img_array)

    # Extract class name and append list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Normalize pixel intensity to range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Construct training and testing splits
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=(1 - config_FERC.TRAIN_RATIO), random_state=42)

valX, testX, valY, testY = train_test_split(testX, testY,
                                            test_size=(config_FERC.TEST_RATIO /
                                                       (config_FERC.TEST_RATIO + config_FERC.VAL_RATIO)),
                                            random_state=42)

# Initialize image data generator
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

# Compile model
opt = SGD(lr=config_FERC.MIN_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=len(config_FERC.CLASSES))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Check if attempt to find optimal learning rate
if config_FERC.LR_FIND:
    print('[INFO] finding learning rate')
    # Initialize learning rate finder in range [1e-10, 1e+1]
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=config_FERC.BATCH_SIZE), 1e-10, 1e+1,
             stepsPerEpoch=np.ceil((len(trainX) / float(config_FERC.BATCH_SIZE))),
             batchSize=config_FERC.BATCH_SIZE)

    # Plot loss against learning rate
    lrf.plot_loss()
    plt.savefig(config_FERC.LRFIND_PLOT_PATH)

    # Exit script
    sys.exit(0)

# If learning rate range is known, initialize cyclic learning rate method
stepSize = config_FERC.STEP_SIZE * (trainX.shape[0] // config_FERC.BATCH_SIZE)
clr = CyclicLR(mode=config_FERC.CLR_METHOD, base_lr=config_FERC.MIN_LR, max_lr=config_FERC.MAX_LR, step_size=stepSize)

# Train model
H = model.fit(x=aug.flow(trainX, trainY, batch_size=config_FERC.BATCH_SIZE),
              validation_data=(valX, valY),
              steps_per_epoch=trainX.shape[0] // config_FERC.BATCH_SIZE,
              epochs=config_FERC.NUM_EPOCHS,
              callbacks=[clr],
              verbose=1)

# Save model
model.save('GoogLeNet_FERC.model', save_format='h5')

# Evaluate model and print classification report
preds = model.predict(x=testX, batch_size=config_FERC.BATCH_SIZE)
report = classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=config_FERC.CLASSES)
t = open(config_FERC.REPORT_PATH, 'w')
t.write(report)
t.close()

# Construct training plot
N = np.arange(0, config_FERC.NUM_EPOCHS)
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
plt.savefig(config_FERC.TRAINING_PLOT_PATH)

# Construct learning rate plot
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config_FERC.CLR_PLOT_PATH)
