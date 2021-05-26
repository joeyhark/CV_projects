from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense
from tensorflow.keras import backend as K


class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation='relu', weightsPath=None):
        # Initialize model and input shape
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # Account for image format with channels first (Theano, Pytorch)
        if K.image_data_format() == 'channels_first':
            inputShape = (numChannels, imgRows, imgCols)

        # Construct 1st CONV -> RELU -> POOL instance
        model.add(Conv2D(20, 5, padding='same', input_shape=inputShape))  # 20=filter amount, 5=filter size 5x5
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Construct 2nd CONV -> RELU -> POOL instance
        model.add(Conv2D(50, 5, padding='same'))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Construct 1st FC -> ACTIVATION instance
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # Construct 2nd FC -> ACTIVATION instance
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        # Load existing weights from trained model if applicable
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
