import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CapsLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kwargs):
        super(CapsLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsules, self.num_capsules, self.input_dim_capsules, self.dim_capsules],
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        inputs_expand = tf.expand_dims(inputs, 2)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1])
        inputs_hat = tf.einsum('bijh,ijhd->bijd', inputs_tiled, self.W)

        b = tf.zeros_like(inputs_hat[:, :, :, 0])
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

    def get_config(self):
        config = super(CapsLayer, self).get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iters": self.routing_iters,
        })
        return config
