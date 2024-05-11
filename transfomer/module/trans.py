import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.fc1 = tf.keras.layers.Dense(d_hidden, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, d_hidden, num_heads, dropout=0.1):
        super(Encoder, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_hidden)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)


class Transformer(tf.keras.Model):
    def __init__(self, d_model, d_input, d_channel, d_output, d_hidden, num_heads, num_layers, dropout=0.1, pe=False):
        super(Transformer, self).__init__()

        self.encoders_1 = [Encoder(d_model, d_hidden, num_heads, dropout) for _ in range(num_layers)]
        self.encoders_2 = [Encoder(d_model, d_hidden, num_heads, dropout) for _ in range(num_layers)]

        self.embedding_channel = tf.keras.layers.Dense(d_model)
        self.embedding_input = tf.keras.layers.Dense(d_model)

        self.gate = tf.keras.layers.Dense(2, activation='softmax')
        self.output_layer = tf.keras.layers.Dense(d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def build_positional_encoding(self, length, d_model):
        pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))

        # Sinusoidal positional encoding for even indices
        pe_sin = tf.sin(pos * div_term)

        # Cosine positional encoding for odd indices
        pe_cos = tf.cos(pos * div_term)

        # Interleave sin and cos positional encodings
        pe = tf.reshape(tf.concat([tf.expand_dims(pe_sin, 2), tf.expand_dims(pe_cos, 2)], axis=2), [length, d_model])

        return pe[tf.newaxis, ...]

    def positional_encoding(self,input_shape):
        positions = tf.range(input_shape[0], dtype=tf.float32)[:, tf.newaxis]
        d_model = input_shape[1]
        # angle_rads = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] / tf.math.pow(10000, (
        #         2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] / tf.math.pow(10000.0, (
                2 * (tf.range(d_model, dtype=tf.float32) // 2)) / tf.cast(d_model, tf.float32))

        sines = tf.math.sin(positions * angle_rads[:, 0::2])
        cosines = tf.math.cos(positions * angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return pos_encoding

    def call(self, x, training=None):
        # Step-wise
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        if self.pe:
            pe = self.build_positional_encoding(tf.shape(encoding_1)[1], self._d_model)
            encoding_1 += pe

            # pe = self.positional_encoding((self._d_input, self._d_model))

        for encoder in self.encoders_1:
            encoding_1 = encoder(encoding_1, training)

        # Channel-wise
        encoding_2 = self.embedding_input(tf.transpose(x, [0, 2, 1]))
        channel_to_gather = encoding_2

        for encoder in self.encoders_2:
            encoding_2 = encoder(encoding_2, training)

        # encoding_1 = tf.reshape(encoding_1, [encoding_1.shape[0], -1])
        # encoding_2 = tf.reshape(encoding_2, [encoding_2.shape[0], -1])
        encoding_1_shape = tf.shape(encoding_1)
        encoding_2_shape = tf.shape(encoding_2)

        encoding_1 = tf.reshape(encoding_1, [encoding_1_shape[0], encoding_1_shape[1] * encoding_1_shape[2]])
        encoding_2 = tf.reshape(encoding_2, [encoding_2_shape[0], encoding_2_shape[1] * encoding_2_shape[2]])

        concatenated_encodings = tf.concat([encoding_1, encoding_2], axis=-1)
        gate = self.gate(concatenated_encodings)

        encoding = tf.concat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], axis=-1)
        output = self.output_layer(encoding)

        return output
