import tensorflow as tf
from tf.keras import layers


class WaveBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias=False):
        super(WaveBlock, self).__init__()

        self.filter = layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  activation='tanh')

        self.gate = layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  activation='sigmoid')
        self.conv1x1 = layers.Conv1D(filters=filters,
                                     kernel_size=1)

    def call(self, inputs):

        x = self.filter(inputs) * self.gate(inputs)
        x = self.conv1x1(x)
        return x

class WaveNet(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rates,
                 use_bias=False):
        super(WaveNet, self).__init__()

        self.blocks = [WaveBlock(filters, kernel_size, dr, use_bias)
                       for dr in dilation_rates]

        self.conv1 = layers.Conv1D(filters,1)
        # this is basically like applying
        # a fully connected network to each filter in previous layer
        self.conv2 = layers.Conv1D(256,1)

    def call(self, inputs, training=False, mask=None):
        # each timestep should be one-hot encoded
        # and then an encoding should be calculated
        # -> each timestep has a series of features
        x = inputs
        outputs = 0
        for block in self.blocks:
            conv, out = block(x)
            outputs += out
            x += conv

        outputs = layers.Activation('relu')(outputs)
        outputs = self.conv1(outputs)
        outputs = layers.Activation('relu')(outputs)
        outputs = self.conv2(outputs)
        # softmax over filters dimension
        softmax = tf.keras.activations.softmax(outputs,
                                               axis=-1)
        return softmax








