from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Inputs, BatchNormalization, Flatten, Dense, MaxPooling2D, UpSampling2D

def get_Encoder(Layers, Hidden_Channels, Starting_Channels):
    inputs = Inputs(shape = 64,64,3)

    x = inputs
    channels = Starting_Channels

    for l in range(Layers-1):
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x),
        x = MaxPooling2D()(x)
        channels = channels / 2

    x = Conv2D(Hidden_Channels*2, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x),
    x = Conv2D(Hidden_Channels, 3, padding = 'same')(x)

    Encoder = keras.Model(inputs, x)
    return Encoder

def Decoder():

def Model():