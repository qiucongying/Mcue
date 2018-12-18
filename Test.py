#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:06:13 2017
Author: Yibin Huang, Congying Qiu

Paper: Surface Defect Saliency of Magnetic Tile
Preprint: https://www.researchgate.net/publication/325882869_Surface_Defect_Saliency_of_Magnetic_Tile
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import glob
import cv2
import math
import time

from model import UNet
from MCueSal2 import Mcue2

# parameter configurations
batch_size = 8
ROISize = 196
num_class = 2,
num_channels = 3
imWidth, imHeight = 98, 98
img_shape = [imHeight, imWidth]


def boundarydiscard(img, bwidth, dtimes):
    '''
    Discard the bondaries for input image
    '''
    w, h = img.shape[:2]
    img[0:bwidth, :] = 0
    img[w-bwidth:w, :] = 0
    img[:, 0:bwidth] = 0
    img[:, h-bwidth:h] = 0

    return img


def cut(inp, low, high):
    return min(high, max(inp, low))


def GetROI(image, i, j):
    Cellx = cut(i*ROISize, 0, imw-ROISize-1)
    Celly = cut(j*ROISize, 0, imh-ROISize-1)
    ROI = image[Celly:Celly+ROISize, Cellx:Cellx+ROISize]

    return ROI


# model configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

label = tf.placeholder(tf.int32, shape=[None]+img_shape)
bboxlabel = tf.placeholder(tf.float32, shape=[None]+[4])


# define model
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape +
                                [num_channels], num_class=num_class)
    img = model.input
    [conv10, bboxPred] = model.output
    pred = tf.clip_by_value(conv10, -10.0, 10.0)

saver = tf.train.Saver()  # must be added in the end

''' Main '''
init_op = tf.global_variables_initializer()
sess.run(init_op)
with sess.as_default():
    # restore from a checkpoint if exists
    try:
        load_from_checkpoint = './Trainlog3F_bs'+str(batch_size)+'/'
        module_file = tf.train.latest_checkpoint(load_from_checkpoint)
        saver.restore(sess, module_file)

    except:
        print ('unable to load checkpoint!')
        sys.exit(0)
    # debug
    tic = time.time()
    imgs = glob.glob('./data/img/'+"/Imgs/*."+'jpg')

    for imgname in imgs:
        src = cv2.imread(imgname)
        height, width = src.shape[:2]
        if height < ROISize:
            src = cv2.resize(src, (int(width*ROISize/height),
                                   ROISize), interpolation=cv2.INTER_NEAREST)
        if width < ROISize:
            src = cv2.resize(
                src, (ROISize, int(height*ROISize/width)), interpolation=cv2.INTER_NEAREST)

        # Retrive ROI from Mcue2 and regenerate a new one by designed height and weight
        Mcue_img = Mcue2(src[:, :, 0])
        src[:, :, 1] = Mcue_img
        imh, imw = src.shape[:2]
        numc = math.ceil(imw / ROISize)
        numr = math.ceil(imh / ROISize)
        Salimg = np.zeros((imh, imw), dtype='float32')

        for i in range(numc):
            for j in range(numr):
                ROI = GetROI(src, i, j)
                ROI = cv2.resize(ROI, (imHeight, imWidth),
                                 interpolation=cv2.INTER_NEAREST)
                Cellx = cut(i*ROISize, 0, imw-ROISize-1)
                Celly = cut(j*ROISize, 0, imh-ROISize-1)
                if len(ROI.shape) == 2:
                    ROI = ROI[None, :, :, None]
                else:
                    ROI = ROI[None, :, :, :]

                pred_logits = sess.run([pred], feed_dict={img: ROI})
                pred_logits = np.array(pred_logits[0][0])
                pred_map = np.argmax(pred_logits, axis=2)
    #            cv2.imshow("ROI",ROI[0])
                Mfe = 1 / \
                    (np.exp((pred_logits[:, :, 0]-pred_logits[:, :, 1])*0.8)+1)
                Mfe = boundarydiscard(Mfe, 1, 200)
                Mfe = cv2.resize(Mfe, (ROISize, ROISize))
                ROI = Salimg[Celly:Celly+ROISize, Cellx:Cellx+ROISize]
                cat2 = np.concatenate(
                    (Mfe[:, :, None], ROI[:, :, None]), axis=2)
                ROI = np.max(cat2, axis=2)
                Salimg[Celly:Celly+ROISize, Cellx:Cellx+ROISize] = ROI
                Salimg[Celly:Celly+ROISize, Cellx:Cellx+ROISize] = Mfe + ROI

        if height < imh or width < imw:
            Salimg = cv2.resize(Salimg, (width, height))

        # write the image to output file
        Salimg = np.clip(Salimg, 0, 1)
        Salimg = (Salimg*255).astype('uint8')
        Salname = imgname.split('.jpg')[0]
        Salname = Salname.replace('Imgs', 'Saliency')
        Salname = Salname+'_bs'+str(batch_size)+'.png'
        cv2.imwrite(Salname, Salimg)

        src = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
        rsimg = np.concatenate((src, Salimg), axis=1)
        cv2.imshow("rsimg", rsimg)
        cv2.waitKey(1)

toc = time.time()
used = (toc-tic)/len(imgs)
print("each Used %.2f ms", used)
cv2.destroyAllWindows()
