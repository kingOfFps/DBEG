import time
from keras.layers import *
from keras.models import *

from main.model.embedding import *
from main.utils.dataProcessing import *
from main.utils.plot import *
from main.utils.utils import *
from main.model.attention import *
from main.model.tcn import *





def stdAttn(params, input_shape):
    """这个是中期论文用的模型"""
    inputs = Input(shape=input_shape)
    """第一个分支（对不同特征维度做注意力计算）：位置编码+attention+tcn(空洞时间卷积网络)"""
    pos_encoding = positional_encoding_3d(inputs.shape[1], inputs.shape[2])
    x1 = inputs + pos_encoding
    x1 = kerasAttention(x1,params)
    x1 = Bidirectional(LSTM(params['units'], return_sequences=True))(x1)
    # x1 = lstm_layer(x1,params['units'])
    """第二个分支（对不同时间步做注意力计算）：attention+tcn(空洞时间卷积网络)"""
    x2 = tf.transpose(inputs, [0, 2, 1])
    x2 = kerasAttention(x2,params)
    x2 = tf.transpose(x2, [0, 2, 1])
    x2 = Bidirectional(LSTM(params['units'], return_sequences=True))(x2)
    # x2 = lstm_layer(x2,params['units'])
    """通过将两个分支得到的矩阵拼接后送入Dense中，获得大小为2的网络层，作为后期两个矩阵相加的权重"""
    concatenated = tf.keras.layers.Concatenate(axis=1)([x1, x2])
    gate_weights = Dense(2, activation='softmax')(Flatten()(concatenated))
    """使用上面Dense训练得到的两个权重将两个分支的信息结合起来，结合有两种方式，一是直接相加
    二是矩阵拼接，例如x_combined = tf.keras.layers.Concatenate(axis=1)([w1, w2])，
    这里用的是第一种"""
    w1 = gate_weights[:, 0:1, tf.newaxis] * x1
    w2 = gate_weights[:, 1:2, tf.newaxis] * x2
    if w1.shape[1] == w2.shape[1]:
        x_combined = w1 + w2
    else:
        x_combined = tf.concat([w1, w2], axis=1)
    """此外，两个分支计算所得的矩阵 对应的权重，除了可以用大小为2的Dense训练得到，也可以通过如下方式得到：
    a = tf.Variable(initial_value=0.5, trainable=True, constraint=lambda t: tf.clip_by_value(t, 0, 1))
    x_combined = a*x1_tcn + (1-a)*x2_tcn"""
    """将x_combined展平后送入大小为1的Dense，作为时间序列预测回归任务的输出。因为是回归，推荐：activation='sigmoid'"""
    x_out = Flatten()(x_combined)
    # output = Dense(1, activation='sigmoid')(x_out)
    output = Dense(1)(x_out)
    model = Model(inputs=inputs, outputs=output)
    return model


def featureBranch(params, input_shape):
    """
    params = {'epoch': 50, 'units': 64, 'stepin': 32, 'stepout': 1, 'batchsize': 16}
    """
    inputs = Input(shape=input_shape)
    """特征分支（对不同特征维度做注意力计算）：位置编码+attention+tcn(空洞时间卷积网络)"""
    pos_encoding = positional_encoding_3d(inputs.shape[1], inputs.shape[2])
    x1 = inputs + pos_encoding
    # x1 = attention_block(x1, way='multiply')
    x1 = attention_block11(x1, params,way='multiply')
    x1 = Bidirectional(LSTM(params['units'], return_sequences=True))(x1)
    """将x_combined展平后送入大小为1的Dense，作为时间序列预测回归任务的输出。因为是回归，推荐：activation='sigmoid'"""
    x_out = Flatten()(x1)
    output = Dense(1)(x_out)
    model = Model(inputs=inputs, outputs=output)
    return model


def timeBranch(params, input_shape):
    """
    params = {'epoch': 50, 'units': 64, 'stepin': 32, 'stepout': 1, 'batchsize': 16}
    """
    """这个是中期论文用的模型，和v4的区别：把tcn换成了bilstm"""
    inputs = Input(shape=input_shape)
    """时间分支（对不同时间步做注意力计算）：attention+tcn(空洞时间卷积网络)"""
    x2 = tf.transpose(inputs, [0, 2, 1])
    # x2 = attention_block(x2,way='multiply')
    x2 = attention_block11(x2, params,way='multiply')
    x2 = tf.transpose(x2, [0, 2, 1])
    x2 = Bidirectional(LSTM(params['units'], return_sequences=True))(x2)

    x_out = Flatten()(x2)
    output = Dense(1)(x_out)
    model = Model(inputs=inputs, outputs=output)
    return model

def noGate(params, input_shape):
    """这个是中期论文用的模型，和v4的区别：把tcn换成了bilstm"""
    inputs = Input(shape=input_shape)
    """第一个分支（对不同特征维度做注意力计算）：位置编码+attention+tcn(空洞时间卷积网络)"""
    pos_encoding = positional_encoding_3d(inputs.shape[1], inputs.shape[2])
    x1 = inputs + pos_encoding
    x1 = attention_block(x1, way='multiply')
    x1 = Bidirectional(LSTM(params['units']//4, return_sequences=True))(x1)
    # x1 = lstm_layer(x1,params['units'])

    """第二个分支（对不同时间步做注意力计算）：attention+tcn(空洞时间卷积网络)"""
    x2 = tf.transpose(inputs, [0, 2, 1])
    x2 = attention_block(x2, way='multiply')
    x2 = tf.transpose(x2, [0, 2, 1])
    x2 = Bidirectional(LSTM(params['units']//4, return_sequences=True))(x2)
    # x2 = lstm_layer(x2,params['units'])

    if x1.shape[2] == x2.shape[2]:
        x_combined = x1 + x2
    else:
        x_combined = tf.concat([x1, x2], axis=2)

    """将x_combined展平后送入大小为1的Dense，作为时间序列预测回归任务的输出。因为是回归，推荐：activation='sigmoid'"""
    x_out = Flatten()(x_combined)
    # output = Dense(1, activation='sigmoid')(x_out)
    output = Dense(1)(x_out)
    model = Model(inputs=inputs, outputs=output)
    return model


def lstm_layer(x, units):
    x = LSTM(units, return_sequences=True)(x)
    # x = LSTM(units // 2, return_sequences=True)(x)
    return x