import tensorflow as tf

class CapsLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kwargs):
        super(CapsLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iters = routing_iters

    def build(self, input_shape):
        # (batch_size, 1152, 8)
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]
        # W: [6*6*32=1152(输入胶囊数), 10(输出胶囊数), 8输入维度, 16输出维度]
        self.W = self.add_weight(
            shape=[self.input_num_capsules, self.num_capsules, self.input_dim_capsules, self.dim_capsules],
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # (batch, 6*6*32, 8) → (batch, 6*6*32, 1, 8)
        inputs_expand = tf.expand_dims(inputs, 2)
        # ->(batch, 6*6*32, 10, 8)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1])
        # u_hat_ij = W_ij · u_i → 预测输出(batch_size,6*6*32, 10, 16)
        inputs_hat = tf.einsum('bijh,ijhd->bijd', inputs_tiled, self.W)

        b = tf.zeros_like(inputs_hat[:, :, :, 0])
        # return output[batch_size , 10, 16]
        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)
            outputs = self.squash(tf.reduce_sum(c[..., tf.newaxis] * inputs_hat, axis=1))
            if i < self.routing_iters - 1:
                b += tf.reduce_sum(inputs_hat * outputs[:, tf.newaxis, :, :], axis=-1)
        return outputs

    def squash(self, s, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + 1e-9)
        return scale * s
