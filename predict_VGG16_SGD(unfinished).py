from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import random
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Hyperparameters, input dimensions, and paths
IMAGE_DIMS = (96, 96, 3)
INPUT_DATASET = 'FERC_sorted'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
REPORT_PATH = 'VGGNet16_report'
BS = 32

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
labels = np.argmax(labels, axis=1)

# Construct training and testing splits
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=(1 - TRAIN_RATIO), random_state=42)

valX, testX, valY, testY = train_test_split(testX, testY,
                                            test_size=(TEST_RATIO / (TEST_RATIO + VAL_RATIO)), random_state=42)

# Construct image data generator object
valAug = ImageDataGenerator()

# Add mean subtraction to image generator
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
valAug.mean = mean

# Load model
model = load_model('VGG16_SGD.model')

# Evaluate accuracy
predIxs = model.predict(x=valAug.flow(testX, batch_size=BS), steps=(len(testX) // BS) + 1)
predIxs = np.argmax(predIxs, axis=1)
report = classification_report(testY, predIxs, target_names=lb.classes_)
t = open(REPORT_PATH, 'w')
t.write(report)
t.close()
