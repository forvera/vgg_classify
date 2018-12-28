import os
import tensorflow as tf
import datetime
import numpy as np
from nets.vgg.vgg_16 import vgg_16
from nets.vgg.tools import loss, optimize, accuracy
from config.config import cfg
from utils.preprocessor import BatchPreprocessor

def train():
    num_epochs = cfg.NUM_EPOCHS
    batch_size = cfg.BATCH_SIZE
    learning_rate = cfg.LEARNING_RATE
    num_classes = cfg.NUM_CLASSES
    dropout_keep_prob = cfg.DROPOUT_KEEP_PROB
    txt_file_path = cfg.TXT_FILE_PATH
    train_image_path = cfg.TRAIN_IMAGE_PATH
    val_image_path = cfg.VAL_IMAGE_PATH
    trainable = cfg.TRAINABLE
    ckpt_path = cfg.CKPT_PATH

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    model = vgg_16(trainable, num_classes, dropout_keep_prob)

    predict = model.build(x, reuse=False)

    train_processor = BatchPreprocessor(os.path.join(txt_file_path, 'train.txt'), train_image_path, num_classes, shuffle=True)
    test_processor = BatchPreprocessor(os.path.join(txt_file_path, 'val.txt'), val_image_path, num_classes, shuffle=True)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_processor.labels)/ batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(test_processor.labels) / batch_size).astype(np.int16)

    # with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(ckpt_path):
            variables = tf.contrib.framework.get_variables_to_restore()
            variables_to_restore = [v for v in variables if v.name.split('/')[1]!='fc8']
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, ckpt_path)

        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1
            while step < train_batches_per_epoch:
                batch_x, batch_y = train_processor.next_batch(batch_size)



    pass


if __name__ == '__main__':
    train()