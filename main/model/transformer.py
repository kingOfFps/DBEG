import keras
from keras import Input, Model
from keras.layers import *



def transformer(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units,
                dropout=0.0,mlp_dropout=0.0, ):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    # outputs = layers.Dense(1, activation="softmax")(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs
    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


if __name__ == '__main__':
    stepin = 55
    feature = 5
    input_shape = (stepin, feature)
    model = transformer(input_shape, head_size=256, num_heads=1, ff_dim=4, num_transformer_blocks=4,
                        mlp_units=[128],dropout=0.25, mlp_dropout=0.4, )
