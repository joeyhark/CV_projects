import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from LRF_and_Architectures.smaller_VGGNet3 import SmallerVGGNet2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

matplotlib.use('Agg')

# Hyperparameters, input dimensions, and paths
EPOCHS = 200
INIT_LR = 1e-3  # Default value for Adam
BS = 32
IMAGE_DIMS = (96, 96, 3)
INPUT_DATASET = 'FERC_sorted'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MODEL_PATH = 'smaller_VGGNet3_FERC.model'
PLOT_PATH = 'smaller_VGGNet3_FERC.png'

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
model = SmallerVGGNet2.build(numChannels=IMAGE_DIMS[2], imgRows=IMAGE_DIMS[0],
                            imgCols=IMAGE_DIMS[1], numClasses=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train model
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(valX, valY),
              steps_per_epoch=len(trainX) / BS,
              epochs=EPOCHS,
              verbose=1)

# Save model
model.save(MODEL_PATH, save_format='h5')

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
plt.savefig(PLOT_PATH)
