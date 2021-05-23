import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from LRF_and_Architectures.smaller_VGGNet import SmallerVGGNet
from LRF_and_Architectures.learning_rate_finder import LearningRateFinder
from LRF_and_Architectures.clr_callback import CyclicLR
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import sys

matplotlib.use('Agg')

# Hyperparameters, input dimensions, and paths
EPOCHS = 200
MIN_LR = 1e-5
MAX_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
INPUT_DATASET = 'FERC_sorted'
LRFIND_PLOT_PATH = "smaller_VGGNet_SGD_LRF_plot.png"
REPORT_PATH = "smaller_VGGNet_SGD_report"
STEP_SIZE = 8
CLR_METHOD = "triangular"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
LR_FIND = False

# Initialize preprocessed data and label lists
data = []
labels = []

# Extract and shuffle image paths
imagePaths = sorted(list(paths.list_images(INPUT_DATASET)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # Load image, resize, convert to array, and append list
    image = cv2.imread(imagePath)
    resized = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
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
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=(1 - TRAIN_RATIO), random_state=42)

valX, testX, valY, testY = train_test_split(testX, testY,
                                            test_size=(TEST_RATIO / (TEST_RATIO + VAL_RATIO)), random_state=42)

# Construct image data generator object
aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

# Compile model
model = SmallerVGGNet.build(numChannels=IMAGE_DIMS[2], imgRows=IMAGE_DIMS[0],
                            imgCols=IMAGE_DIMS[1], numClasses=len(lb.classes_))
opt = SGD(lr=MIN_LR, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Check if attempt to find optimal learning rate
if LR_FIND:
    print('[INFO] finding learning rate')
    # Initialize learning rate finder in range [1e-10, 1e+1]
    lrf = LearningRateFinder(model)
    lrf.find(aug.flow(trainX, trainY, batch_size=BS), 1e-10, 1e+1,
             stepsPerEpoch=np.ceil((len(trainX) / float(BS))),
             batchSize=BS)

    # Plot loss against learning rate
    lrf.plot_loss()
    plt.savefig(LRFIND_PLOT_PATH)

    # Exit script
    sys.exit(0)

# If learning rate range is known, initialize cyclic learning rate method
stepSize = STEP_SIZE * (trainX.shape[0] // BS)
clr = CyclicLR(mode=CLR_METHOD, base_lr=MIN_LR, max_lr=MAX_LR, step_size=stepSize)

# Train model
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(valX, valY),
              steps_per_epoch=len(trainX) / BS,
              epochs=EPOCHS,
              verbose=1)

# Save model
model.save('smaller_VGGNet_FERC_SGD.model', save_format='h5')

plt.style.use('ggplot')
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper left')
plt.savefig('smaller_VGGNet_FERC_SGD.png')
