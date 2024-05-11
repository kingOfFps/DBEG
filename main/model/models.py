import keras.backend as K
from keras.layers import Input, Conv1D, Bidirectional, Multiply
from keras.layers import LSTM
from keras.layers.core import *
from keras.layers.merging import concatenate
from keras.models import *
from keras.layers import *

from .attention import *
from .tcn import *
from main.model.embedding import *


def cnn_bilstm_attention(params, input_shape):
    """CNN-BiLSTM-Attention
    最佳参数：way = add，activation='sigmoid'"""
    dropout_rate = 0
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=params['units'], kernel_size=1)(inputs)
    # x = Conv1D(filters=params['units'], kernel_size=1, activation='relu', padding='causal')(inputs)
    x = Dropout(dropout_rate)(x)
    # BiLSTM效果比LSTM更好
    # lstm_out = Bidirectional(LSTM(params['units']//4, return_sequences=True))(x)
    lstm_out = LSTM(params['units'], return_sequences=True)(x)
    # lstm_out = Dropout(dropout_rate)(lstm_out)
    # x = attention_block(lstm_out, way='add')
    x = attention_block11(lstm_out, params,way='add')

    attention_mul = Flatten()(x)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def cnn_attention(params, input_shape):
    """CNN-BiLSTM-Attention
    最佳参数：way = add，activation='sigmoid'
    中期设置：用经典的Attention,而不是用自己的attention"""
    dropout_rate = 0
    inputs = Input(shape=input_shape)
    x = inputs
    x = Conv1D(filters=params['units'], kernel_size=1)(x)
    # x = Attention()([x,x,x])
    # x= attention_block(x)
    x= attention_block11(x,params)
    # x = Conv1D(filters=params['units'], kernel_size=1, activation='relu', padding='causal')(inputs)
    # x = Dropout(dropout_rate)(x)
    # x = attention_block(x, way='add')
    attention_mul = Flatten()(x)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def tcnAttention(params, input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    x = attention_block(x)
    # x = ProbSparseSelfAttention(temperature=1)(x)
    x = LSTM(params['units'], return_sequences=True)(x)
    x = tcn_layer(x=x)
    # x = Bidirectional(LSTM(params['units'], return_sequences=True))(x)
    x = attention_block(x)
    # 固定的Flatten与全连接层    x = attention_block(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


