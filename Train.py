#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 7 16:33:04 2018
Author: Yibin Huang, Congying Qiu

Paper: Surface Defect Saliency of Magnetic Tile
Preprint: https://www.researchgate.net/publication/325882869_Surface_Defect_Saliency_of_Magnetic_Tile
"""
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED=0
import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.set_random_seed(SEED)

import os
import cv2
import time
import random

from model import UNet
from utils2 import  mean_IU
from matplotlib import pyplot as plt

# save and compute metrics

class MyMCue():
    def __init__(self,
            learning_rate=0.0001,
            lr_decay=0.9,
            mode='last', #'last' and init and zero
            batch_size=8,
#            checkpoint_path='./Trainlog3F_bs'+str(batch_size)+'/',
            load_from_checkpoint='./Trainlog3/',
            num_channels=3,
            num_class=2,
            namelist_name='namelist.npy',
            srcWidth=196, srcHeight=196,
            imWidth=98, imHeight=98,
            gtWidth=98, gtHeight=98,
            tot_iter=44000):
        
        self.learning_rate=learning_rate
        self.lr_decay=lr_decay
        self.mode=mode
        self.batch_size=batch_size
        self.checkpoint_path='./Trainlog3F_bs'+str(self.batch_size)+'/'
        self.load_from_checkpoint=load_from_checkpoint
        self.num_channels=num_channels
        self.num_class=num_class
        
        self.srcWidth, self.srcHeight=srcWidth, srcHeight
        self.imWidth, self.imHeight=imWidth, imHeight
        self.gtWidth, self.gtHeight=gtWidth, gtHeight
        self.tot_iter=tot_iter
        
        self.namelist=np.load(namelist_name)
        self.lenth=self.namelist.shape[0]-1
        self.iter_epoch=int(self.lenth/batch_size)

        self.img_shape=[imHeight,imWidth]
        self.gt_shape=[gtHeight,gtWidth]


    #################
    # Preprocessing
    #################
    def getCrop(self, src,ceny,cenx):        
        imHeight,imWidth=self.imHeight, self.imWidth
        height,width=src.shape[:2] 
        
        if height<imHeight:
            times=float(imHeight)/float(height)
            src=cv2.resize(src,( int(float(width)*times+2), imHeight), interpolation=cv2.INTER_NEAREST)
            cenx=int(cenx*times)
            ceny=int(ceny*times)        
            height, width=src.shape[:2]
            
        if width<imWidth:
            times=float(imWidth)/float(width)
            src=cv2.resize(src,(imWidth, int(float(height)*times +2) ), interpolation=cv2.INTER_NEAREST)
            cenx=int(cenx*times)
            ceny=int(ceny*times)
     
        lef,top=ceny-int(imWidth/2), cenx-int(imWidth/2)
        rig,bot=lef+imWidth, top+imHeight
    
        if lef<0:
            rig=rig-lef
            lef=0        
        if top<0:
            bot=bot-top
            top=0
        if rig>src.shape[1]:
            lef=src.shape[1]-imWidth
            rig=src.shape[1]
        if bot>src.shape[0]:
            top=src.shape[0]-imHeight
            bot=src.shape[0]
        ROI=src[top:bot,lef:rig]
        
        return ROI
     
        
    def calArea(y_batch):
        output=np.mean(y_batch.astype('float32'), axis=0)*100.0
        return output
    
    
    def calBbox(y_batch) :
        bbox=np.zeros((y_batch.shape[0],4))
        for i in range(y_batch.shape[0]):
            bwimg=y_batch[i]>0             
            output=cv2.connectedComponentsWithStats(bwimg.astype(np.uint8), 8, cv2.CV_32S)
            stats=output[2]
            centroids=output[3]
            num_labels=output[0]
            area=np.sum(y_batch[i])
    
            minid = np.argmin(np.abs(stats[:][4]-area))
            
            if num_labels==1:
                bbox[i]=[y_batch.shape[1]/2,y_batch.shape[2]/2,0,0]
            else:
                bbox[i]=[centroids[minid][0],centroids[minid][1],stats[minid][2],stats[minid][3]]
            
        return bbox
        
    
    def nextbatch(self, it):
        # build index and read images
        start_idx=it % (self.lenth-self.batch_size-2)
        end_idx=start_idx+self.batch_size
        for idx in range(start_idx, end_idx): 
            image=cv2.imread(self.namelist[idx])
            image=image.astype(np.float32)
    
            lbname=self.namelist[idx].replace('_train','_label')
            gt=cv2.imread(lbname,0)/225
            gt=gt.astype(np.uint8) 
     
            # computes the connected components labeled image and also produces a statistics output for each label
            ## in our case, the surface defect inspection, the max connected component is image background
            ### the rest connected components are ROIs
            output=cv2.connectedComponentsWithStats(gt, 4, cv2.CV_32S)
            num_labels=output[0]
            stats=output[2]        
            centroids=output[3]  
    
            if num_labels<2:
                rx,ry=random.randint(0, self.srcWidth),random.randint(0, self.srcWidth)
                ROI= self.getCrop(image,rx,ry)
                labelROI=self.getCrop(gt,rx,ry)
    
            else:    
                # pick up image background
                areas=stats[:][4]
                maxid=np.argmax(areas)
                
                # random offet for centroids of all ROIs               
                ROI=np.zeros_like(image)
                labelROI=np.zeros_like(gt)
                for i in range(num_labels):
                    if i != maxid:
                        rx,ry=np.random.randint(10),np.random.randint(10)
                        rx,ry=np.random.randint(10),np.random.randint(10)
                        ROI+=self.getCrop(image,int(centroids[i][0]+rx-5),int(centroids[i][1]+ry-5))
                        labelROI+=self.getCrop(gt,int(centroids[i][0]+rx-5),int(centroids[i][1]+ry-5))
    
            # generate ROI batch 
            img_batch=np.zeros((self.batch_size, self.imHeight, self.imWidth, self.num_channels), np.float32)
            label_batch=np.zeros((self.batch_size, self.gtHeight, self.gtWidth), np.int32)
            
            img_batch[idx-start_idx, :, :, :]=ROI
            label_batch[idx-start_idx, :, :]=labelROI
    
        return img_batch,label_batch


    def training(self, train_bbox=False):
        iter_epoch=int(self.lenth/self.batch_size)
        
        # define input holders
        label=tf.placeholder(tf.int32, shape=[None]+self.img_shape)
        bboxlabel=tf.placeholder(tf.float32, shape=[None]+[4])
        
        # define model
        with tf.name_scope('unet'):
            model=UNet().create_model(img_shape=self.img_shape+[self.num_channels],
                                      num_class=self.num_class)
            img=model.input
            [conv10 ,bboxPred]=model.output
            pred=tf.clip_by_value(conv10, -10.0, 10.0) 
        
        # define loss    
        with tf.name_scope('cross_entropy'):
            cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred)
            label2=tf.reshape(label,[-1, self.imHeight, self.imWidth])
            weight=tf.add(label2,1)
            weight=tf.cast(weight, tf.float32)
            cross_entropy=tf.multiply(weight,cross_entropy)
            cross_entropy_loss=tf.reduce_mean(cross_entropy)
            
        with tf.name_scope('Euclidean_loss'):
            diff=tf.subtract(bboxlabel, bboxPred)
            dist=tf.multiply(diff,diff)
            Euclidean_distance_loss=tf.reduce_mean(dist)        
        
        # define optimizer
        global_step=tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('learning_rate'):
            learning_rate=tf.train.exponential_decay(self.learning_rate, global_step,
                                                   iter_epoch, self.lr_decay, staircase=True)
        train_Unet=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)
        Train_bbox=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Euclidean_distance_loss, global_step=global_step)
        
        # compute dice score for simple evaluation during training
        ''' Tensorboard visualization '''
        # cleanup pervious info
        if self.load_from_checkpoint == '':
            cf=os.listdir(self.checkpoint_path)
            for item in cf: 
                if 'event' in item: 
                    os.remove(os.path.join(self.checkpoint_path, item))
                    
        # define summary for tensorboard
        tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
        tf.summary.scalar('Euclidean_loss', Euclidean_distance_loss)
        
        summary_merged=tf.summary.merge_all()
       
        # configuration session       
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess=tf.Session(config=config)

        train_writer=tf.summary.FileWriter(self.checkpoint_path, sess.graph)
        saver=tf.train.Saver() # must be added in the end
        
        ''' Main '''
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        sess.graph.finalize()
        
        with sess.as_default():
            # Load the pre-trained process
            if self.mode == 'last':
                try:
                    module_file=tf.train.latest_checkpoint(self.checkpoint_path)    
                    saver.restore(sess, module_file)                  
                except:
                    print ('unable to load checkpoint ...' )
                    
            elif self.mode == 'init':
                try:
                    saver.restore(sess,  self.load_from_checkpoint+'model-8013')
                    print ('--> load from checkpoint '+ self.load_from_checkpoint)
                except:
                    print ('unable to load checkpoint ...' )   
            
            start=global_step.eval()

            # Main Training Loop
            for it in range(start, self.tot_iter):
                if it %  10000 == 0 and it > start:       
                    saver.save(sess,  self.checkpoint_path+'model', global_step=global_step)
                    print ('save a checkpoint at '+  self.checkpoint_path+'model-'+str(it))
                    
                x_batch, y_batch=self.nextbatch(it)                        
         
                bbox=self.calBbox(y_batch)
                feed_dict={ img: x_batch, label: y_batch, bboxlabel:bbox }
                
                if train_bbox:
                    _, loss, loss2, summary, lr, pred_logits, bboxp=sess.run([Train_bbox,  
                                            cross_entropy_loss, 
                                            Euclidean_distance_loss,
                                            summary_merged,
                                            learning_rate,
                                            pred,
                                            bboxPred
                                            ], feed_dict=feed_dict)
                   
                else :
                    _, loss, loss2, summary, lr, pred_logits, bboxp=sess.run([train_Unet, 
                                            cross_entropy_loss, 
                                            Euclidean_distance_loss,
                                            summary_merged,
                                            learning_rate,
                                            pred,
                                            bboxPred
                                            ], feed_dict=feed_dict)
           
                if it % 200 == 0 : 
                    pred_map=np.argmax(pred_logits[0],axis=2)
                    score, _=mean_IU(pred_map, y_batch[0])

                    train_writer.add_summary(summary, it)
                    print ('[iter %d, epoch %.3f]: lr=%f loss=%f,loss2=%f, mean_IU=%f' % (it, float(it)/ iter_epoch, lr, loss, loss2,score))
                
                if it % 20 == 0:
                    # save the image with bounding box 
                    print(time.strftime( '%Y-%m-%d %X', time.gmtime( time.time( ))))
                    pred_map=np.argmax(pred_logits[0],axis=2)            
                    imgout= np.concatenate((y_batch[0],pred_map),axis=1)
                    imgout= np.concatenate((x_batch[0,:,:,1],imgout*255.0),axis=1)
                    imgout= np.concatenate((x_batch[0,:,:,0],imgout),axis=1)
                    imgout=imgout.astype('uint8')
                    imgout=cv2.cvtColor(imgout, cv2.COLOR_GRAY2BGR)
        			
                    bbox0=[bbox[0][0]-bbox[0][2]/2-5,bbox[0][1]-bbox[0][3]/2-5,bbox[0][0]+bbox[0][2]/2+5,bbox[0][1]+bbox[0][3]/2+5]
                    bbox0=(np.array(bbox0)).astype('int32')
                    imgout= cv2.rectangle(imgout,(bbox0[0],bbox0[1]),(bbox0[2],bbox0[3]),(255,55,0),2)
                    bbox1=[bboxp[0][0]-bboxp[0][2]/2-5,bboxp[0][1]-bboxp[0][3]/2-5,bboxp[0][0]+bboxp[0][2]/2+5,bboxp[0][1]+bboxp[0][3]/2+5]
                    bbox1=(np.array(bbox1)).astype('int32')
                    imgout= cv2.rectangle(imgout,(bbox1[0],bbox1[1]),(bbox1[2],bbox1[3]),(0,55,255),2)
                    plt.imshow(imgout)           
                    plt.show()            

        saver.save(sess, self.checkpoint_path+'model', global_step=global_step)
             
