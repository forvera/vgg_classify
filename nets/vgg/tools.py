import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def loss(logits, batch_y=None):
    return tf.reduce_mean(tf.nn.softmax_cross_entrop_with_logits(logits=logits, labels=batch_y))

def optimize(loss, learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)

def accuracy(logits, y):
    logits = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acc

def ckpt2npy(file_path):
    vgg_16 = 'vgg_16'
    reader = pywrap_tensorflow.NewCheckpointReader(file_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    layers = ['conv1/conv1_1', 'conv1/conv1_2', 'conv2/conv2_1', 'conv2/conv2_2','conv3/conv3_1', 'conv3/conv3_2','conv3/conv3_3', 'conv4/conv4_1', 'conv4/conv4_2','conv4/conv4_3','conv5/conv5_1','conv5/conv5_2', 'conv5/conv5_3', 'fc6', 'fc7']
    data = {
        'conv1_1': [],
        'conv1_2': [],
        'conv2_1': [],
        'conv2_2': [],
        'conv3_1': [],
        'conv3_2': [],
        'conv3_3': [],
        'conv4_1': [],
        'conv4_2': [],
        'conv4_3': [],
        'conv5_1': [],
        'conv5_2': [],
        'conv5_3': [],
        'fc6': [],
        'fc7': []
    }
 
    for op_name in layers:
 
        biases_variable = reader.get_tensor(vgg_16+'/'+op_name+'/biases')
        weights_variable = reader.get_tensor(vgg_16+'/'+op_name+'/weights')
        tmp={'biases':biases_variable,'weights':weights_variable}
        data[op_name.split('/')[-1]] = tmp
    return data

if __name__ == '__main__':

    file_path = './pretrained/vgg_16.ckpt'
    data = ckpt2npy(file_path)
    print('end')