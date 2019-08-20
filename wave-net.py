import tensorflow as tf
from tensorflow.keras import layers


class WaveBlock(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dilation_rate,
                 use_bias=False, **kwargs):
        super(WaveBlock, self).__init__(**kwargs)

        self.filter = layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  padding='causal',
                                  activation='tanh')

        self.gate = layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  padding='causal',
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

        self.blocks = [WaveBlock(filters, kernel_size, dr, use_bias, name='block{}'.format(i))
                       for i,dr in enumerate(dilation_rates)]

        self.conv1 = layers.Conv1D(filters, 1, name='conv_out1')
        # this is basically like applying
        # a fully connected network to each filter in previous layer
        self.conv2 = layers.Conv1D(256, 1, name='conv_out2')

    @tf.function
    def call(self, inputs, training=False, mask=None):
        # each timestep should be one-hot encoded
        # and then an encoding should be calculated
        # -> each timestep has a series of features
        x = inputs
        outputs = 0
        for i in range(len(self.blocks)):
            conv = self.blocks[i](x)
            outputs += conv
            x += conv

        outputs = layers.Activation('relu')(outputs)
        outputs = self.conv1(outputs)
        outputs = layers.Activation('relu')(outputs)
        outputs = self.conv2(outputs)
        # softmax over filters dimension
        softmax = tf.keras.activations.softmax(outputs,
                                               axis=-1)
        return softmax

if __name__ == "__main__":
    import numpy as np
    from datetime import datetime

    # Set up logging.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/wave-net/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().

    data = np.random.rand(5000).reshape((10,100,5))

    net = WaveNet(5, 2,
                  [1,2,4])

    #net.compile(tf.optimizers.Adam(), tf.losses.categorical_crossentropy)
    # fails if other python interpreter is opened!! - it takes a lot of gpu memory for some reason
    # https://stackoverflow.com/questions/34514324/error-using-tensorflow-with-gpu
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf.summary.trace_on(graph=True, profiler=True)
    result = net(data)
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)

    print(net.summary())







