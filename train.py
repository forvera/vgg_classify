import os
import tensorflow as tf
import datetime
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

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    model = vgg_16('./pretrained/vgg_16/vgg_16.ckpt', trainable, dropout_keep_prob)

    saver = tf.train.Saver()

    train_processor = BatchPreprocessor(os.path.join(txt_file_path, 'train.txt'), train_image_path, num_classes, shuffle=True)
    test_processor = BatchPreprocessor(os.path.join(txt_file_path, 'val.txt'), val_image_path, num_classes, shuffle=True)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_processor.labels.size / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(test_processor.labels.size / batch_size).astype(np.int16)

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            sess.run(tf.globel_variables_initializer())
            for epoch in range(num_epochs):
                print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
                step = 1
                while step < train_batches_per_epoch:
                    batch_x, batch_y = train_processor.next_batch(batch_size)



    pass


if __name__ == '__main__':
    train()