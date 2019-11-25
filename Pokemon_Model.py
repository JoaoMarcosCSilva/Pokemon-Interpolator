%tensorflow_version 2.x
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, MaxPooling2D, UpSampling2D

def get_Encoder(Layers, Hidden_Channels, Starting_Channels):
    
    inputs = Input(shape = (64,64,3))
    x = inputs

    channels = Starting_Channels
    
    for l in range(Layers-1):
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        channels = int(channels / 2)

    x = Conv2D(Hidden_Channels*2, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(Hidden_Channels, 3, padding = 'same')(x)

    Encoder = keras.Model(inputs, x)
    
    print("Latent code's shape:", Encoder.output_shape)

    return Encoder

def get_Decoder(Layers, Hidden_Shape, Encoder_Starting_Channels):

    inputs = Input(shape = (Hidden_Shape))
    x = inputs
    
    x = Conv2D(Hidden_Shape[-1]*2, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    
    channels = int(Encoder_Starting_Channels / (2**(Layers-1)))

    for l in range(Layers-1):
        channels = channels * 2

        x = UpSampling2D()(x)
        x = Conv2D(channels, 3, activation = 'relu', padding = 'same')(x)
        x = BatchNormalization()(x)

    x = Conv2D(3, 3, activation = 'sigmoid', padding = 'same') (x)

    Decoder = keras.Model(inputs, x)

    print("Output image's shape: ", Decoder.output_shape)

    return Decoder

def get_Model (Layers, Hidden_Channels, Starting_Channels):
    inputs = Input(shape = (64,64,3))
    Encoder = get_Encoder(Layers, Hidden_Channels, Starting_Channels)(inputs)
    Decoder = get_Decoder(Layers, Encoder.output_shape[1:], Starting_Channels)(Encoder)
    Model = keras.Model(inputs, Decoder)
    return Encoder, Decoder, Model