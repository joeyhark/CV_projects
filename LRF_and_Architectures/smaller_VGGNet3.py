from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense


class SmallerVGGNet2:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation='relu'):
        # Initialize model and input shape
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)
        channelDim = -1

        # Construct 1st CONV -> RELU -> POOL instance
        model.add(Conv2D(32, 3, padding='same', input_shape=inputShape))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # Construct 2nd CONV -> RELU -> POOL instance (CONV -> RELU x2)
        model.add(Conv2D(64, 3, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(Conv2D(64, 3, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Construct 3nd CONV -> RELU -> POOL instance (CONV -> RELU x2)
        model.add(Conv2D(128, 3, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(Conv2D(128, 3, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Construct 4nd CONV -> RELU -> POOL instance (CONV -> RELU x2)
        model.add(Conv2D(256, 3, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(Conv2D(256, 3, padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Construct 1st FC -> ACTIVATION instance
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation(activation))
        model.add(BatchNormalization(axis=channelDim))
        model.add(Dropout(0.5))

        # Construct 2nd FC -> ACTIVATION instance
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        return model
