import keras.backend as K
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.keras.engine.base_layer import Layer
from math import sqrt

from main.utils.masking import *
import math
from tensorflow import keras
from keras.layers.core import *
from keras.layers.merging import concatenate
from keras import Input, Model, layers
from keras.layers import *


def qkv_attention(query, key, value):
    """计算注意力权重并返回输出"""
    matmul_qk = tf.matmul(query, key, transpose_b=True)  # QK^T

    # 缩放 matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # softmax归一化得到注意力权重
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # 加权聚合值
    output = tf.matmul(attention_weights, value)

    return output


def attention_block(inputs, way='multiply'):
    """
    注意力机制：简单的点积计算方式。通过全连接层和softmax函数计算注意力权重，并将权重与输入张量在
    最后一个维度上进行拼接，得到最终的输出。
    :param inputs.shape = (batch_size, time_steps, input_dim)
    :return:返回inputs经过注意力计算后的结果
    """
    input_dim = int(inputs.shape[2])
    """这一步相当于计算查询向量（query）和键向量（key）的点积，并进行归一化，得到注意力权重（attention weight）"""
    x = inputs
    x = Dense(input_dim, activation='softmax')(x)
    waySet = {'concat', 'multiply', 'add'}
    assert way in waySet
    if way == 'concat':
        x = concatenate([inputs, x])
    elif way == 'multiply':
        x = Multiply()([inputs, x])
    elif way == 'add':
        x += inputs
    return x


def attention_block1(inputs, way='multiply', use_scale=True):
    """
    改进的注意力机制
    """
    input_dim = int(inputs.shape[2])
    """这一步相当于计算查询向量（query）和键向量（key）的点积，并进行归一化，得到注意力权重（attention weight）"""
    x = inputs
    # 缩放 matmul_qk
    x = tf.matmul(x, x, transpose_b=True)  # QK^T
    x = Dense(input_dim)(x)
    """是否进行缩放"""
    if use_scale:
        depth = tf.cast(tf.shape(x)[-1], tf.float32)
        x = x / tf.math.sqrt(depth)

    # softmax归一化得到注意力权重
    x = tf.nn.softmax(x, axis=-1)
    waySet = {'concat', 'multiply', 'add'}
    assert way in waySet
    if way == 'concat':
        x = concatenate([inputs, x])
    elif way == 'multiply':
        x = Multiply()([inputs, x])
    elif way == 'add':
        x += inputs
    return x

def kerasAttention(x,params):
    d_model = params['d_model']
    # 假设query、key、value的原始维度都是dim
    query = Dense(d_model)(x)  # 把query映射到128维
    key = Dense(d_model)(x)  # 把key映射到128维
    value = Dense(d_model)(x)  # 把value映射到128维
    x = Attention()([query,value, key])
    return x

def attention_block11(inputs, params,way='multiply', use_scale=True):
    """     改进的注意力机制    """
    input_dim = int(inputs.shape[2])
    d_model = params['d_model']
    x = inputs
    # 缩放 matmul_qk
    q = x
    k = x
    v = Dense(d_model)(x)

    qk = tf.matmul(q, k, transpose_b=True)
    # qk = Multiply()([q, k])
    """是否进行缩放"""
    if use_scale:
        depth = tf.cast(tf.shape(inputs)[-1], tf.float32)
        qk = qk / tf.math.sqrt(depth)
    # softmax归一化得到注意力权重
    x = Dense(d_model, activation='softmax')(qk)
    waySet = {'concat', 'multiply', 'add'}
    assert way in waySet
    if way == 'concat':
        x = concatenate([v, x])
    elif way == 'multiply':
        x = Multiply()([v, x])
    elif way == 'add':
        x += v
    return x

def attention_block22(inputs, params,way='multiply', use_scale=True):
    """
    改进的注意力机制
    """
    input_dim = int(inputs.shape[2])
    d_model = params['d_model']
    x = inputs
    # 缩放 matmul_qk
    q = x
    k = x
    v = Dense(d_model)(x)

    qk = tf.matmul(q, k, transpose_b=True)
    # qk = Multiply()([q, k])
    """是否进行缩放"""
    if use_scale:
        depth = tf.cast(tf.shape(inputs)[-1], tf.float32)
        qk = qk / tf.math.sqrt(depth)
    # softmax归一化得到注意力权重
    x = tf.nn.softmax(qk, axis=-1)
    waySet = {'concat', 'multiply', 'add'}
    assert way in waySet
    if way == 'concat':
        x = concatenate([v, x])
    elif way == 'multiply':
        x = tf.matmul(x, v)
    elif way == 'add':
        x += v
    return x


def attention_block2(inputs, single_attention_vector=False):
    """
    注意力计算：通过对输入张量进行转置、全连接层和softmax函数计算得到注意力权重，
    然后将权重与输入张量进行逐元素相乘，得到最终的输出
    如果single_attention_vector参数为True，则会对注意力权重进行平均处理，这可能会导致信息的损失。

    针对attention_block2的改进策略可以有以下几点：
    考虑引入多头注意力机制，以增强模型的表达能力。
    在计算注意力权重时，可以引入位置编码或关系编码等信息，以增强不同维度之间的关系。
    对于single_attention_vector参数为True的情况，可以考虑使用其他方式对注意力权重进行处理，以避免信息的损失
    :param inputs.shape = (batch_size, time_steps, input_dim)
    :return:返回inputs经过注意力计算后的结果
    """
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def attention_block3(inputs):
    """
    注意力计算（加入位置编码）
    :param inputs: 输入张量，shape为(batch_size, time_steps, input_dim)
    :return: 输出张量，shape为(batch_size, time_steps, input_dim)
    """
    input_dim = int(inputs.shape[2])
    x = inputs
    # 计算查询向量和键向量的点积，并进行归一化，得到注意力权重
    x = Dense(input_dim, activation='softmax')(x)

    # 添加位置编码
    time_steps = int(inputs.shape[1])
    position = np.arange(time_steps)[:, np.newaxis]
    div_term = np.exp(np.arange(0, input_dim, 2) * -(math.log(10000.0) / input_dim))
    pos_encoding = np.zeros((time_steps, input_dim))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    pos_encoding = pos_encoding[np.newaxis, ...]

    # 将位置编码与注意力权重相乘
    x = x * tf.constant(pos_encoding, dtype=tf.float32)

    # 将注意力权重与输入张量相乘，并返回结果
    output_attention_mul = tf.multiply(inputs, x)
    return output_attention_mul


class ProbSparseSelfAttention(keras.layers.Layer):
    def __init__(self, temperature, **kwargs):
        super(ProbSparseSelfAttention, self).__init__(**kwargs)
        self.temperature = temperature

    def build(self, input_shape):
        # 初始化权重矩阵
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], input_shape[-1]),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs):
        # 计算相似度得分
        similarity = tf.matmul(inputs, tf.transpose(inputs, [0, 2, 1]))

        # 归一化得分
        attention_weights = tf.nn.softmax(similarity / self.temperature, axis=-1)

        # 加权求和
        output = tf.matmul(attention_weights, inputs)

        return output


class FullAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = tf.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            # https://stackoverflow.com/questions/47447272/does-tensorflow-have-the-function-similar-to-pytorchs-masked-fill
            num = 3.4 * math.pow(10, 38)
            scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        A = self.dropout(tf.keras.activations.softmax(scale * scores, axis=-1))
        V = tf.einsum("bhls,bshd->blhd", A, values)

        return V


class ProbAttention(Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = tf.broadcast_to(tf.expand_dims(K, -3), (B, H, S, L, E))

        indx_q_seq = tf.random.uniform((S,), maxval=L, dtype=tf.int32)
        indx_k_seq = tf.random.uniform((sample_k,), maxval=L, dtype=tf.int32)

        K_sample = tf.gather(K_expand, tf.range(S), axis=2)

        K_sample = tf.gather(K_sample, indx_q_seq, axis=2)
        K_sample = tf.gather(K_sample, indx_k_seq, axis=3)

        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.einsum("...ij->...ji", K_sample)))
        # find the Top_k query with sparisty measurement
        M = tf.math.reduce_max(Q_K_sample, axis=-1) - tf.raw_ops.Div(x=tf.reduce_sum(Q_K_sample, axis=-1), y=L)
        M_top = tf.math.top_k(M, n_top, sorted=False)[1]
        batch_indexes = tf.tile(tf.range(Q.shape[0])[:, tf.newaxis, tf.newaxis], (1, Q.shape[1], n_top))
        head_indexes = tf.tile(tf.range(Q.shape[1])[tf.newaxis, :, tf.newaxis], (Q.shape[0], 1, n_top))

        idx = tf.stack(values=[batch_indexes, head_indexes, M_top], axis=-1)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(Q, idx)

        Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2]))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = tf.reduce_sum(V, -2)
            contex = tf.identity(tf.broadcast_to(tf.expand_dims(V_sum, -2), [B, H, L_Q, V_sum.shape[-1]]))
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = tf.math.cumsum(V, axis=-1)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            num = 3.4 * math.pow(10, 38)
            scores = (scores * attn_mask.mask) + (-((attn_mask.mask * num + num) - num))

        attn = tf.keras.activations.softmax(scores, axis=-1)  # nn.Softmax(dim=-1)(scores)
        batch_indexes = tf.tile(tf.range(V.shape[0])[:, tf.newaxis, tf.newaxis], (1, V.shape[1], index.shape[-1]))
        head_indexes = tf.tile(tf.range(V.shape[1])[tf.newaxis, :, tf.newaxis], (V.shape[0], 1, index.shape[-1]))

        idx = tf.stack(values=[batch_indexes, head_indexes, index], axis=-1)

        context_in = tf.tensor_scatter_nd_update(context_in, idx, tf.matmul(attn, V))

        return tf.convert_to_tensor(context_in)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        queries = tf.reshape(queries, (B, H, L, -1))
        keys = tf.reshape(keys, (B, H, S, -1))
        values = tf.reshape(values, (B, H, S, -1))

        U = self.factor * np.ceil(np.log(S)).astype('int').item()
        u = self.factor * np.ceil(np.log(L)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, u, U)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L)
        # update the context with selected top_k queries
        context = self._update_context(context, values, scores_top, index, L)

        return context


class AttentionLayer(Layer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.d_model = d_model

        self.inner_attention = attention
        self.query_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.key_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.value_projection = tf.keras.layers.Dense(d_values * n_heads)
        self.out_projection = tf.keras.layers.Dense(d_model)
        self.n_heads = n_heads

    def build(self, input_shape):
        print(input_shape)
        B, L, _ = input_shape[0]
        _, S, _ = input_shape[1]
        H = self.n_heads

        self.queries = self.add_weight(shape=(B, L, H, self.d_model),
                                       initializer='random_normal',
                                       trainable=True)

        self.keys = self.add_weight(shape=(B, S, H, self.d_model),
                                    initializer='random_normal',
                                    trainable=True)

        self.values = self.add_weight(shape=(B, S, H, self.d_model),
                                      initializer='random_normal',
                                      trainable=True)

    def call(self, inputs, attn_mask=None):
        queries, keys, values = inputs
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        self.queries = tf.reshape(self.query_projection(queries), (B, L, H, -1))
        self.keys = tf.reshape(self.key_projection(keys), (B, S, H, -1))
        self.values = tf.reshape(self.value_projection(values), (B, S, H, -1))

        out = tf.reshape(self.inner_attention([self.queries, self.keys, self.values], attn_mask=attn_mask), (B, L, -1))

        return self.out_projection(out)


if __name__ == '__main__':
    attn = AttentionLayer(ProbAttention(False), 128, 4)
    queries = tf.zeros((32, 20, 128))
    keys = tf.zeros((32, 20, 128))
    values = tf.zeros((32, 20, 128))
    print(attn([queries, keys, values]))
