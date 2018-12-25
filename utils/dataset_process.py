# -*- coding: utf-8 -*- 

import numpy as np 
import cv2
import os


def main():
    data_path = os.getcwd()
    im_paths = os.path.join(data_path, 'data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/')
    im_output_paths = os.path.join(data_path, 'data/VOC2012_classification_dataset/Images/' )
    txt_output_paths = os.path.join(data_path, 'data/VOC2012_classification_dataset/train_val_txt/')
    txt_paths = os.path.join(data_path, 'data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/')
    # paths = '../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
    # ImgOutput_path = '../data/VOC2012_classification_dataset/Images/'
    # txtOutput_path = '../data/VOC2012_classification_dataset/train_val_txt/'
    # txt_paths = '../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/'
    cls = [('aeroplane', 0),('bicycle', 1),('bird', 2),('boat', 3),('bottle', 4),('bus', 5),
        ('car', 6),('cat', 7),('chair', 8),('cow', 9),('diningtable', 10),('dog', 11),('horse', 12),
        ('motorbike', 13),('person', 14),('pottedplant', 15),('sheep', 16),('sofa', 17),('train', 18),('tvmonitor', 19)]
    image_name_train = []
    image_labels_train = []
    image_name_val = []
    image_labels_val = []
    for i in range(len(cls)):
        txt_train_paths = os.path.join(txt_paths, cls[i][0]+'_train.txt')
        txt_val_paths = os.path.join(txt_paths, cls[i][0]+'_val.txt')
        fp = open(txt_train_paths, 'r')
        lines = fp.readlines()
        for line in lines:
            items = line.split()
            if int(items[1]) == 1:
                image_name_train.append(items[0] + '.jpg')
                image_labels_train.append(cls[i][1])
        fp.close()
        fp = open(txt_val_paths, 'r')
        lines = fp.readlines()
        for line in lines:
            items = line.split()
            if int(items[1]) == 1:
                image_name_val.append(items[0]+'.jpg')
                image_labels_val.append(cls[i][1])
        fp.close()

    # for i in range(len(image_name_train)):
    #     img = cv2.imread(os.path.join(im_paths, image_name_train[i]))
    #     cv2.imwrite(os.path.join(im_output_paths, 'train', image_name_train[i]), img)
    #     print(i)
    # for i in range(len(image_name_val)):
    #     img = cv2.imread(os.path.join(im_paths, image_name_val[i]))
    #     cv2.imwrite(os.path.join(im_output_paths, 'val', image_name_val[i]), img)
    #     print(i)

    fp = open(os.path.join(txt_output_paths, 'train.txt'), 'w')
    for i in range(len(image_name_train)):
        line = image_name_train[i] + ' ' + str(image_labels_train[i]) + '\n'
        fp.write(line)
    fp.close()
    fp = open(os.path.join(txt_output_paths, 'val.txt'), 'w')
    for i in range(len(image_name_val)):
        line = image_name_val[i] + ' ' + str(image_labels_val[i]) + '\n'
        fp.write(line)
    fp.close()
    # print(len(image_name_train))
    # print(len(image_name_val))


if __name__=='__main__':
    main()
