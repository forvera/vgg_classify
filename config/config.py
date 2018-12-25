from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.TRAIN = edict()

__C.NUM_EPOCHS = 1000
__C.LEARNING_RATE = 1e-4
__C.DROPOUT_KEEP_PROB = 0.5
__C.NUM_CLASSES = 10
__C.CKPT_PATH = './ckpt/model.ckpt'
__C.BATCH_SIZE = 64
__C.TXT_FILE_PATH = './data/VOC2012_classification_dataset/train_val_txt'
__C.TRAIN_IMAGE_PATH = './data/VOC2012_classification_dataset/Images/train'
__C.VAL_IMAGE_PATH = './data/VOC2012_classificaiton_dataset/Images/val'
__C.WEIGHT_DECAY = 0.0005
__C.TRAINABLE = True