import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def unet_conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x


def unet(input_shape, use_cnnt=False, num_layers=3):
    inputs = Input(shape=input_shape)
    x = inputs

    # Downsampling path
    conv_blocks = []
    for i in range(num_layers):
        x = unet_conv_block(x, 32 * 2**i)
        conv_blocks.append(x)
        x = MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = unet_conv_block(x, 32 * 2**num_layers)

    # Upsampling path
    for i in reversed(range(num_layers)):
        if use_cnnt:
            x = Conv2DTranspose(32 * 2**i, (3, 3), strides=(2, 2), padding='same')(x)
        else:
            x = UpSampling2D((2, 2))(x)
        x = concatenate([conv_blocks[i], x])
        x = unet_conv_block(x, 32 * 2**i)

    # Final convolution
    x = Conv2D(1, (1, 1), padding='same')(x)
    return Model(inputs, x)

