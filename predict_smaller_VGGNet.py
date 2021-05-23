from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import random
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Hyperparameters, input dimensions, and paths
EPOCHS = 200
INIT_LR = 1e-3  # Default value for Adam
BS = 32
IMAGE_DIMS = (96, 96, 3)
INPUT_DATASET = 'FERC_sorted'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SINGLE_IMG_TEST = True
PREDICT_ALL = False
TEST_IMAGE = 'test/happiness/test_06.jpg'
REPORT_PATH = 'results/smaller_VGGNet3_FERC_report'
MODEL_PATH = 'results/smaller_VGGNet3_FERC.model'

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

# Load model
model = load_model(MODEL_PATH)

if PREDICT_ALL:
    # Make predictions
    predIxs = model.predict(testX, batch_size=BS, verbose=1)
    predIxs = np.argmax(predIxs, axis=1)
    report = classification_report(testY, predIxs, target_names=lb.classes_)
    t = open(REPORT_PATH, 'w')
    t.write(report)
    t.close()

if SINGLE_IMG_TEST:
    # Load image, resize, normalize, and convert to array
    image = cv2.imread(TEST_IMAGE)
    output = image.copy()
    resized = cv2.resize(output, (96, 96))
    normalized = resized.astype('float') / 255.0
    img_array = img_to_array(normalized)
    image = np.expand_dims(img_array, axis=0)

    # Predict test image
    prob = model.predict(image)[0]
    predIdx = np.argmax(prob)
    label_pred = lb.classes_[predIdx]

    # Test prediction against ground truth
    label = TEST_IMAGE.split(os.path.sep)[-2]
    correct = 'correct' if label == label_pred else 'incorrect'

    # Construct and display output
    text = '{}: {:.2f}% ({})'.format(label_pred, prob[predIdx] * 100, correct)
    output = imutils.resize(output, width=400)
    cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Output', output)
    cv2.waitKey(0)
