from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from LeNet_MNIST import LeNet

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

# One-hot encode labels
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

# Initialize model and optimizer
opt = SGD(lr=0.01)
Model = LeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10)
Model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train model
Model.fit(trainData, trainLabels, batch_size=128, epochs=20, verbose=1)

# Evaluate accuracy
loss, accuracy = Model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print('[INFO] Loss = {} \n Accuracy = {}'.format(loss, accuracy))

# Save weights
Model.save_weights('LeNet_weights_MNIST.hdf5')
