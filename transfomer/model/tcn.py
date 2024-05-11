from keras.layers.core import *
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense, Conv1D, Activation, Add, Input, Conv1D, \
    Bidirectional, Multiply

def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0):
    """
    Defines the residual block for the WaveNet TCN
    """
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding=padding)(x)
        x = Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = Add()([prev_x, x])
    return res_x, res_x


def tcn_layer(x,  nb_filters=64, kernel_size=2, nb_stacks=1, dilations=None,
              padding='causal', use_skip_connections=True, dropout_rate=0):
    """
    Creates a TCN model for sequence modeling and forecasting.
    """
    if dilations is None:
        dilations = [1, 2, 4, 8, 16]
    # Apply convolutional blocks
    skip_connections = []
    for s in range(nb_stacks):
        for d in dilations:
            x, skip_out = residual_block(x, dilation_rate=d, nb_filters=nb_filters,
                                         kernel_size=kernel_size, padding=padding,
                                         dropout_rate=dropout_rate)
            skip_connections.append(skip_out)
    if use_skip_connections:
        x = Add()(skip_connections)
    x = Activation('relu')(x)
    return x