import tensorflow as tf
from nets.vgg.tools import ckpt2npy

class vgg_16():
    
    def __init__(self, num_classes, trainable=True, keep_prob=0.5):
        self.num_classes = num_classes
        self.trainable = trainable
        self.keep_prob = keep_prob

    def build(self, rgb_x):
        """
        load variable from npy to build the VGG
        :param x: x image [batch, height, width, 3] values scaled [0, 1]
        """
        VGG_MEAN = [103.939, 116.779, 123.68]
        rgb_scaled = rgb_x * 255.0
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr_x = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]   
        ])
        assert bgr_x.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self._conv_layer(rgb_x, kernels=[3, 3, 3, 64], name='conv1_1', reuse=True)
        self.conv1_2 = self._conv_layer(self.conv1_1, kernels=[3, 3, 64, 64], name='conv1_2', reuse=True)
        self.pool1 = self._max_pool(self.conv1_2, name='pool1')

        self.conv2_1 = self._conv_layer(self.pool1, kernels=[3, 3, 64, 128], name='conv2_1', reuse=True)
        self.conv2_2 = self._conv_layer(self.conv2_1, kernels=[3, 3, 128, 128], name='conv2_2', reuse=True)
        self.pool2 = self._max_pool(self.conv2_2, name='pool2')

        self.conv3_1 = self._conv_layer(self.pool2, kernels=[3, 3, 128, 256], name='conv3_1', reuse=True)
        self.conv3_2 = self._conv_layer(self.conv3_1, kernels=[3, 3, 256, 256], name='conv3_2', reuse=True)
        self.conv3_3 = self._conv_layer(self.conv3_2, kernels=[3, 3, 256, 256], name='conv3_3', reuse=True)
        self.pool3 = self._max_pool(self.conv3_3, name='pool3')

        self.conv4_1 = self._conv_layer(self.pool3, kernels=[3, 3, 256, 512], name='conv4_1', reuse=True)
        self.conv4_2 = self._conv_layer(self.conv4_1, kernels=[3, 3, 512, 512], name='conv4_2', reuse=True)
        self.conv4_3 = self._conv_layer(self.conv4_2, kernels=[3, 3, 512, 512], name='conv4_3', reuse=True)
        self.pool4 = self._max_pool(self.conv4_3, name='pool4')

        self.conv5_1 = self._conv_layer(self.pool4, kernels=[3, 3, 512, 512], name='conv5_1', reuse=True)
        self.conv5_2 = self._conv_layer(self.conv5_1, kernels=[3, 3, 512, 512], name='conv5_2', reuse=True)
        self.conv5_3 = self._conv_layer(self.conv5_2, kernels=[3, 3, 512, 512], name='conv5_3', reuse=True)
        self.pool5 = self._max_pool(self.conv5_3, name='pool5')

        with tf.variable_scope('fatten'):
            shape = self.pool5.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            self.fatten = tf.reshape(self.pool5, [-1, dim])

        self.fc6 = self._fc_layer(self.fatten, 4096, name='fc6', reuse=True)
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)
        if self.trainable:
            self.relu6 = self._dropout_layer(self.relu6, name='fc6_drop', keep_prob=self.keep_prob)

        self.fc7 = self._fc_layer(self.relu6, 4096, name='fc7', reuse=True)
        self.relu7 = tf.nn.relu(self.fc7)
        if self.trainable:
            self.relu7 = self._dropout_layer(self.relu7, name='fc7_drop', keep_prob=self.keep_prob)

        self.fc8 = self._fc_layer(self.relu7, self.num_classes, name='fc8', reuse=True)
        predict = self.fc8
        return predict

    def _avg_pool(self, x, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        return tf.nn.avg_pool(x, ksize=ksize, strides=strides, name=name, padding=padding)

    def _max_pool(self, x, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, name=name, padding=padding)

    def _conv_layer(self, x, name, kernels, activation='relu', strides=[1, 1, 1, 1], padding='SAME', is_training=False, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            kernel = tf.get_variable(name='weights', shape=kernels, initializer=tf.truncated_normal_initializer(0, 1))
            print(kernel.name)
            biases = tf.get_variable(name='biases', shape=kernels[-1], initializer=tf.truncated_normal_initializer(0, 1))
            print(biases.name)
            conv = tf.nn.conv2d(x, kernel, strides=strides, padding=padding)
            bias = tf.nn.bias_add(conv, biases)
            if activation=='relu':
                return tf.nn.relu(bias)
            elif activation==None:
                return bias
            return None

    def _dropout_layer(self, x, name, keep_prob=0.5):
        return tf.nn.dropout(x, keep_prob, name=name)

    def _fc_layer(self, x, nodes, name, reuse=None):
        shape = [x.shape[-1], nodes]
        with tf.variable_scope(name, reuse=reuse):
            weights = tf.get_variable(name='weights', shape=shape, initializer=tf.truncated_normal_initializer(0, 1))
            biases = tf.get_variable(name='biases', shape=shape[-1], initializer=tf.truncated_normal_initializer(0, 1))
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
