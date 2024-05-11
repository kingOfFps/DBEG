import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from keras.layers import LeakyReLU
import math
import numpy as np


def positional_encoding(input_shape):
    positions = tf.range(input_shape[0], dtype=tf.float32)[:, tf.newaxis]
    d_model = input_shape[1]
    angle_rads = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] / tf.math.pow(10000, (
            2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))
    sines = tf.math.sin(positions * angle_rads[:, 0::2])
    cosines = tf.math.cos(positions * angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
    return pos_encoding


class PositionalEmbedding(Layer):
    """
    位置编码：位置编码的目的是将输入序列中的位置信息嵌入到模型中，以便模型能够理解输入序列中不同位置的信息。：
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 用0矩阵初始化位置编码矩阵
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        """用np.arange()生成一个长度为$max_len$的一维向量$position$，并使用np.expand_dims()函数
        将其转换为一个二维矩阵。这个向量表示输入序列中每个位置的索引值。"""
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        """我们使用np.arange()函数生成一个长度为$d_{model}$的一维向量，其中每个元素的值等于$2i/d_{model}$，
        其中$i$表示向量中当前元素的索引。这个向量表示每个位置向量中不同维度的变化速率。"""
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))
        """我们使用np.sin()和np.cos()函数分别计算每个位置向量中偶数维和奇数维的值。具体来说，对于偶数维$i$，
        我们计算$sin(pos/10000^{i/d_{model}})$；对于奇数维$i$，我们计算$cos(pos/10000^{i/d_{model}})$。
        其中$pos$表示当前位置的索引值。我们将计算得到的位置向量按照偶数维和奇数维交替排列，并将其赋值给位置编码矩阵$pe$。
        pe[:, 0::2]表示取出pe第二个维度中的偶数列出来（0，2，4），pe[:, 1::2]表示取出奇数列出来（1，3，5。。。）"""
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        """将位置编码矩阵$pe$转换为TensorFlow张量，并使用tf.expand_dims()函数在第0维添加一个维度，以便后续计算。"""
        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), 0)

    def call(self, inputs, **kwargs):
        return self.pe[:, :inputs.shape[1]]


class TokenEmbedding(Layer):
    """标记编码的目的是将输入序列中的每个标记转换为一个$d_{model}$维的向量表示"""

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = tf.keras.layers.Conv1D(filters=d_model,
                                                kernel_size=3, padding='causal', activation='linear')
        """定义了一个LeakyReLU激活函数   """
        self.activation = LeakyReLU()

    def call(self, inputs, **kwargs):
        x = self.tokenConv(inputs)
        x = self.activation(x)
        return x


class FixedEmbedding(Layer):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = np.zeros((c_in, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, c_in, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

        w[:, 0::2] = np.sin(position * div_term)
        w[:, 1::2] = np.cos(position * div_term)

        w = tf.convert_to_tensor(w)
        tf.stop_gradient(w)
        w = tf.keras.initializers.Constant(w)
        self.emb = tf.keras.layers.Embedding(c_in, d_model, embeddings_initializer=w)

    def call(self, inputs, **kargs):
        embedding = self.emb(inputs)

        return embedding


class TemporalEmbedding(Layer):
    def __init__(self, d_model, embed_type='fixed', data='ETTh'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else tf.keras.layers.Embedding
        # self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        # self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def call(self, inputs, **kargs):
        # minute_x = self.minute_embed(inputs[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        # weekday_x = self.weekday_embed(inputs[:, :, 2])
        month_x = self.month_embed(inputs[:, :, 0])
        day_x = self.day_embed(inputs[:, :, 1])
        hour_x = self.hour_embed(inputs[:, :, 2])
        return hour_x + day_x + month_x
        # return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(Layer):
    """数据编码，整合了之前的时间、位置、标记编码"""

    def __init__(self, c_in, d_model, embed_type='fixed', dropout=0.1, seq_len=96):
        super(DataEmbedding, self).__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, x_mark=None, **kwargs):
        """将之前的3种编码拼接相加，然后送入dropout中，防止过拟合。"""
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding2(Layer):
    """没有时间提取"""

    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding2, self).__init__()
        self.c_in = c_in
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, x_mark=None, **kwargs):
        """将之前的2种编码拼接相加，然后送入dropout中，防止过拟合。"""
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


def get_positional_encoding(seq_len, d_model):
    pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model // 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))

    # Generate sin and cos terms
    sin_term = tf.sin(pos * div_term)
    cos_term = tf.cos(pos * div_term)

    # Interleave sin and cos terms without cutting off
    pos_enc = tf.reshape(tf.stack([sin_term, cos_term], axis=-1), [seq_len, d_model])

    return pos_enc


def positional_encoding_3d(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将sin应用于数组中的偶数索引（indices为0、2、...）; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # 将cos应用于数组中的奇数索引（indices为1、3、...）; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    return pos_encoding
