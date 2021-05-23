import matplotlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Hyperparameters, input dimensions, and paths
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
INPUT_DATASET = 'FERC_sorted'
IMAGE_DIMS = (224, 224, 3)
LR = 1e-4
BS = 32
EPOCHS = 50
SAVE = True

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

# Construct image data generator objects
trainAug = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.15,
                              zoom_range=0.15,
                              horizontal_flip=True,
                              fill_mode='nearest')

valAug = ImageDataGenerator()

# Add mean subtraction to image generators
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# Construct model
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze layers in base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
opt = SGD(lr=LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
H = model.fit(x=trainAug.flow(trainX, trainY, batch_size=BS),
              validation_data=valAug.flow(valX, valY, batch_size=32),
              steps_per_epoch=len(trainX) / BS,
              validation_steps=len(valX) / BS,
              epochs=EPOCHS,
              verbose=1)

# Save model
if SAVE:
    model.save('VGG16_SGD.model', save_format='h5')

# Construct plot
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
plt.savefig('VGG16_SGD.png')
