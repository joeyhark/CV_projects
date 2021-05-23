from LeNet_MNIST import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
from sklearn.metrics import classification_report
import cv2

# Load MNIST dataset
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Reorganize data matrices based on channel ordering
if K.image_data_format == 'channels_first':
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Normalize data to [0, 1] range
trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

# Construct model with weights loaded
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10, weightsPath='LeNet_weights.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Make predictions
predIxs = model.predict(testData, batch_size=128, verbose=1)
predIxs = np.argmax(predIxs, axis=1)
print(classification_report(testLabels, predIxs))

# Choose random digit 0-10
for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    # Predict random sample from test set
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)

    # Extract image
    if K.image_data_format() == 'channels_first':
        image = (testData[i][0] * 255).astype('uint8')
    else:
        image = (testData[i] * 255).astype('uint8')

    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
