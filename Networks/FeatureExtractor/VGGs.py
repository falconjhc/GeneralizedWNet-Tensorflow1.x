# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../')
sys.path.append('../../')


import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
import math

from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock
from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
from Utilities.utils import PrintNetworkVars

# from Networks.FeatureExtractor.Framework import FeatureExtractor


eps = 1e-9
output_filters = 64
print_separater="#########################################################"

def reviseVarNames(var_input,deleteFrom, add='ext_'):
        var_output = {}
        for ii in var_input:
                prefix_pos = ii.name.find(deleteFrom)
                renamed = ii.name[prefix_pos + 1:]
                parafix_pos = renamed.find(':')
                renamed = renamed[0:parafix_pos]
                renamed = add+renamed
                var_output.update({renamed: ii})
        return var_output


class VGGs(object):
    def __init__(self,
                 config, namePrefix, initializer, penalties, networkInfo):
        
        # super().__init__()
        self.config = config
        self.namePrefix = namePrefix
        self.initializer = initializer
        self.penalties = penalties
        
        self.networkInfo = networkInfo
        
        NetworkSelection={'vgg16net': self.VGG16Net,
                          'vgg19net': self.VGG19Net}
        
        self.NetworkImplementation=NetworkSelection[networkInfo.name]
        
        return
    

    def VGG11Net(self, inputImg, 
                 weight_decay=-1,
                 is_training=False,
                 name_prefix='None',
                 reuse=False):

        if is_training:
            print(print_separater)
            print("Training on Vgg-11")
            print(print_separater)

        if is_training:
            keep_prob=0.5
        else:
            keep_prob=1.0


        with tf.variable_scope(name_prefix):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            
            ## block 1
            conv1_1 = relu(batch_norm(x=conv2d(x=inputImg,
                                               output_filters=64,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv1_1'),
                                    is_training=is_training,
                                    scope='bn1_1',
                                    parameter_update_device=self.networkInfo.device))
            pool1 = tf.nn.max_pool(value=conv1_1,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')

            ## block 2
            conv2_1 = relu(batch_norm(x=conv2d(x=pool1, output_filters=128,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv2_1'),
                                    is_training=is_training,
                                    scope='bn2_1',
                                    parameter_update_device=self.networkInfo.device))


            pool2 = tf.nn.max_pool(value=conv2_1,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

            ## block 3
            conv3_1 = relu(batch_norm(x=conv2d(x=pool2, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_1'),
                                    is_training=is_training,
                                    scope='bn3_1',
                                    parameter_update_device=self.networkInfo.device))

            conv3_2 = relu(batch_norm(x=conv2d(x=conv3_1, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_2'),
                                    is_training=is_training,
                                    scope='bn3_2',
                                    parameter_update_device=self.networkInfo.device))

            pool3 = tf.nn.max_pool(value=conv3_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool3')

            ## block 4
            conv4_1 = relu(batch_norm(x=conv2d(x=pool3, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_1'),
                                    is_training=is_training,
                                    scope='bn4_1',
                                    parameter_update_device=self.networkInfo.device))

            conv4_2 = relu(batch_norm(x=conv2d(x=conv4_1, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_2'),
                                    is_training=is_training,
                                    scope='bn4_2',
                                    parameter_update_device=self.networkInfo.device))

            pool4 = tf.nn.max_pool(value=conv4_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool4')

            ## block 5
            conv5_1 = relu(batch_norm(x=conv2d(x=pool4, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_1'),
                                    is_training=is_training,
                                    scope='bn5_1',
                                    parameter_update_device=self.networkInfo.device))

            conv5_2 = relu(batch_norm(x=conv2d(x=conv5_1, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_2'),
                                    is_training=is_training,
                                    scope='bn5_2',
                                    parameter_update_device=self.networkInfo.device))

            pool5 = tf.nn.max_pool(value=conv5_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool5')




            # block 6
            fc6 = tf.reshape(pool5, [self.config.trainParams.batchSize, -1])
            fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                        output_size=4096,
                                        scope="fc6",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)),
                                keep_prob=keep_prob)

            # block 7
            fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                        output_size=4096,
                                        scope="fc7",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)),
                                keep_prob=keep_prob)

            # block 8
            if 'Style' in name_prefix:
                logitsLabel1 = fc(x=fc7,
                                        output_size=len(self.config.datasetConfig.loadedLabel1Vec),
                                        scope="output_label1",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)
            else:
                logitsLabel1 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])

            
            if 'Content' in name_prefix:
                logitsLabel0 = fc(x=fc7,
                                        output_size=len(self.config.datasetConfig.loadedLabel0Vec),
                                        scope="output_label0",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)
            else:
                logitsLabel0 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])
                    
            
            outputLogits = [logitsLabel0, logitsLabel1]
            features=[conv1_1, conv2_1, conv3_2, conv4_2, conv5_2, fc6, fc7]
            
            if not reuse:
                self.varsTrain = [ii for ii in tf.trainable_variables() if name_prefix in ii.name]
                movingMeans=[ii for ii in tf.global_variables() if name_prefix in ii.name and 'moving_mean' in ii.name]
                movingVars=[ii for ii in tf.global_variables() if name_prefix in ii.name and 'moving_variance' in ii.name]
                varsSave=self.varsTrain+movingMeans+movingVars
                varsSave = reviseVarNames(varsSave,'-')
                self.saver = tf.train.Saver(max_to_keep=1, var_list=varsSave)
    
                print(name_prefix + "@" + self.networkInfo.device)
                PrintNetworkVars(name_prefix)
                print(print_separater)
            


            return outputLogits, features






    def VGG16Net(self, inputImg, 
                 weight_decay=-1,
                 is_training=False,
                 name_prefix='None',
                 reuse=False):

        if is_training:
            print(print_separater)
            print("Training on Vgg-16")
            print(print_separater)
        if is_training:
            keep_prob=0.5
        else:
            keep_prob=1.0


        with tf.variable_scope(name_prefix):
            if reuse:
                tf.get_variable_scope().reuse_variables()


            ## block 1
            conv1_1 = relu(batch_norm(x=conv2d(x=inputImg, 
                                               output_filters=64,
                                               kh=3, kw=3,
                                               sh=1, sw=1,
                                               padding='SAME',
                                               parameter_update_device=self.networkInfo.device,
                                               weight_decay=weight_decay,
                                               initializer=self.initializer,
                                               scope='conv1_1'),
                                      is_training=is_training,
                                      scope='bn1_1',
                                      parameter_update_device=self.networkInfo.device))

            conv1_2 = relu(batch_norm(x=conv2d(x=conv1_1, output_filters=64,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv1_2'),
                                    is_training=is_training,
                                    scope='bn1_2',
                                    parameter_update_device=self.networkInfo.device))

            pool1 = tf.nn.max_pool(value=conv1_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')

            ## block 2
            conv2_1 = relu(batch_norm(x=conv2d(x=pool1, output_filters=128,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv2_1'),
                                    is_training=is_training,
                                    scope='bn2_1',
                                    parameter_update_device=self.networkInfo.device))

            conv2_2 = relu(batch_norm(x=conv2d(x=conv2_1, output_filters=128,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv2_2'),
                                    is_training=is_training,
                                    scope='bn2_2',
                                    parameter_update_device=self.networkInfo.device))

            pool2 = tf.nn.max_pool(value=conv2_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

            ## block 3
            conv3_1 = relu(batch_norm(x=conv2d(x=pool2, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_1'),
                                    is_training=is_training,
                                    scope='bn3_1',
                                    parameter_update_device=self.networkInfo.device))

            conv3_2 = relu(batch_norm(x=conv2d(x=conv3_1, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_2'),
                                    is_training=is_training,
                                    scope='bn3_2',
                                    parameter_update_device=self.networkInfo.device))

            conv3_3 = relu(batch_norm(x=conv2d(x=conv3_2, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_3'),
                                    is_training=is_training,
                                    scope='bn3_3',
                                    parameter_update_device=self.networkInfo.device))

            pool3 = tf.nn.max_pool(value=conv3_3,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool3')

            ## block 4
            conv4_1 = relu(batch_norm(x=conv2d(x=pool3, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_1'),
                                    is_training=is_training,
                                    scope='bn4_1',
                                    parameter_update_device=self.networkInfo.device))

            conv4_2 = relu(batch_norm(x=conv2d(x=conv4_1, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_2'),
                                    is_training=is_training,
                                    scope='bn4_2',
                                    parameter_update_device=self.networkInfo.device))

            conv4_3 = relu(batch_norm(x=conv2d(x=conv4_2, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_3'),
                                    is_training=is_training,
                                    scope='bn4_3',
                                    parameter_update_device=self.networkInfo.device))

            pool4 = tf.nn.max_pool(value=conv4_3,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool4')

            ## block 5
            conv5_1 = relu(batch_norm(x=conv2d(x=pool4, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_1'),
                                    is_training=is_training,
                                    scope='bn5_1',
                                    parameter_update_device=self.networkInfo.device))

            conv5_2 = relu(batch_norm(x=conv2d(x=conv5_1, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_2'),
                                    is_training=is_training,
                                    scope='bn5_2',
                                    parameter_update_device=self.networkInfo.device))

            conv5_3 = relu(batch_norm(x=conv2d(x=conv5_2, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_3'),
                                    is_training=is_training,
                                    scope='bn5_3',
                                    parameter_update_device=self.networkInfo.device))

            pool5 = tf.nn.max_pool(value=conv5_3,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool5')

            fidFeature = pool5


            # block 6
            fc6 = tf.reshape(pool5, [self.config.trainParams.batchSize, -1])
            fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                        output_size=4096,
                                        scope="fc6",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)),
                                keep_prob=keep_prob)
            

            # block 7
            fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                          output_size=4096,
                                          scope="fc7",
                                          parameter_update_device=self.networkInfo.device,
                                          weight_decay=weight_decay,
                                          initializer=self.initializer)),
                                keep_prob=keep_prob)
            
            # block 8
            if 'Style' in name_prefix:
                logitsLabel1 = fc(x=fc7,
                                        output_size=len(self.config.datasetConfig.loadedLabel1Vec),
                                        scope="output_label1",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)
            else:
                logitsLabel1 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])

            if 'Content' in name_prefix:
                logitsLabel0 = fc(x=fc7,
                                        output_size=len(self.config.datasetConfig.loadedLabel0Vec),
                                        scope="output_label0",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)
            else:
                logitsLabel0 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])
            
            outputLogits = [logitsLabel0, logitsLabel1]
            features=[conv1_2, conv2_2, conv3_3, conv4_3, conv5_3, fc6, fc7]
        

            if not reuse:
                self.varsTrain = [ii for ii in tf.trainable_variables() if name_prefix in ii.name]
                movingMeans=[ii for ii in tf.global_variables() if name_prefix in ii.name and 'moving_mean' in ii.name]
                movingVars=[ii for ii in tf.global_variables() if name_prefix in ii.name and 'moving_variance' in ii.name]
                varsSave=self.varsTrain+movingMeans+movingVars
                varsSave = reviseVarNames(varsSave,'-')
                self.saver = tf.train.Saver(max_to_keep=1, var_list=varsSave)
    
                print(name_prefix + "@" + self.networkInfo.device)
                PrintNetworkVars(name_prefix)
                print(print_separater)
                
                


            return outputLogits, features, fidFeature



    


    def VGG19Net(self, inputImg, 
                 weight_decay=-1,
                 is_training=False,
                 name_prefix='None',
                 reuse=False):

        if is_training:
            print(print_separater)
            print("Training on Vgg-19")
            print(print_separater)
        if is_training:
            keep_prob=0.5
        else:
            keep_prob=1.0


        with tf.variable_scope(name_prefix):
            if reuse:
                tf.get_variable_scope().reuse_variables()


            ## block 1
            conv1_1 = relu(batch_norm(x=conv2d(x=inputImg, 
                                               output_filters=64,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv1_1'),
                                    is_training=is_training,
                                    scope='bn1_1',
                                    parameter_update_device=self.networkInfo.device))

            conv1_2 = relu(batch_norm(x=conv2d(x=conv1_1, output_filters=64,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv1_2'),
                                    is_training=is_training,
                                    scope='bn1_2',
                                    parameter_update_device=self.networkInfo.device))

            pool1 = tf.nn.max_pool(value=conv1_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')

            ## block 2
            conv2_1 = relu(batch_norm(x=conv2d(x=pool1, output_filters=128,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv2_1'),
                                    is_training=is_training,
                                    scope='bn2_1',
                                    parameter_update_device=self.networkInfo.device))

            conv2_2 = relu(batch_norm(x=conv2d(x=conv2_1, output_filters=128,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv2_2'),
                                    is_training=is_training,
                                    scope='bn2_2',
                                    parameter_update_device=self.networkInfo.device))

            pool2 = tf.nn.max_pool(value=conv2_2,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

            ## block 3
            conv3_1 = relu(batch_norm(x=conv2d(x=pool2, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_1'),
                                    is_training=is_training,
                                    scope='bn3_1',
                                    parameter_update_device=self.networkInfo.device))

            conv3_2 = relu(batch_norm(x=conv2d(x=conv3_1, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_2'),
                                    is_training=is_training,
                                    scope='bn3_2',
                                    parameter_update_device=self.networkInfo.device))

            conv3_3 = relu(batch_norm(x=conv2d(x=conv3_2, output_filters=256,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv3_3'),
                                    is_training=is_training,
                                    scope='bn3_3',
                                    parameter_update_device=self.networkInfo.device))

            conv3_4 = relu(batch_norm(x=conv2d(x=conv3_3, output_filters=256,
                    kh=3, kw=3,
                    sh=1, sw=1,
                    padding='SAME',
                    parameter_update_device=self.networkInfo.device,
                    weight_decay=weight_decay,
                    initializer=self.initializer,
                    scope='conv3_4'),
            is_training=is_training,
            scope='bn3_4',
            parameter_update_device=self.networkInfo.device))

            pool3 = tf.nn.max_pool(value=conv3_4,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool3')

            ## block 4
            conv4_1 = relu(batch_norm(x=conv2d(x=pool3, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_1'),
                                    is_training=is_training,
                                    scope='bn4_1',
                                    parameter_update_device=self.networkInfo.device))

            conv4_2 = relu(batch_norm(x=conv2d(x=conv4_1, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_2'),
                                    is_training=is_training,
                                    scope='bn4_2',
                                    parameter_update_device=self.networkInfo.device))

            conv4_3 = relu(batch_norm(x=conv2d(x=conv4_2, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_3'),
                                    is_training=is_training,
                                    scope='bn4_3',
                                    parameter_update_device=self.networkInfo.device))


            conv4_4 = relu(batch_norm(x=conv2d(x=conv4_3, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv4_4'),
                                    is_training=is_training,
                                    scope='bn4_4',
                                    parameter_update_device=self.networkInfo.device))


            pool4 = tf.nn.max_pool(value=conv4_4,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool4')

            ## block 5
            conv5_1 = relu(batch_norm(x=conv2d(x=pool4, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_1'),
                                    is_training=is_training,
                                    scope='bn5_1',
                                    parameter_update_device=self.networkInfo.device))

            conv5_2 = relu(batch_norm(x=conv2d(x=conv5_1, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_2'),
                                    is_training=is_training,
                                    scope='bn5_2',
                                    parameter_update_device=self.networkInfo.device))

            conv5_3 = relu(batch_norm(x=conv2d(x=conv5_2, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_3'),
                                    is_training=is_training,
                                    scope='bn5_3',
                                    parameter_update_device=self.networkInfo.device))

            conv5_4 = relu(batch_norm(x=conv2d(x=conv5_3, output_filters=512,
                                            kh=3, kw=3,
                                            sh=1, sw=1,
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv5_4'),
                                    is_training=is_training,
                                    scope='bn5_4',
                                    parameter_update_device=self.networkInfo.device))

            pool5 = tf.nn.max_pool(value=conv5_4,
                                ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool5')
            fidFeature=pool5

            # block 6
            fc6 = tf.reshape(pool5, [self.config.trainParams.batchSize, -1])
            fc6 = tf.nn.dropout(x=relu(fc(x=fc6,
                                        output_size=4096,
                                        scope="fc6",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)),
                                keep_prob=keep_prob)

            # block 7
            fc7 = tf.nn.dropout(x=relu(fc(x=fc6,
                                        output_size=4096,
                                        scope="fc7",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)),
                                keep_prob=keep_prob)

            # block 8
            if 'Style' in name_prefix:
                logitsLabel1 = fc(x=fc7,
                                        output_size=len(self.config.datasetConfig.loadedLabel1Vec),
                                        scope="output_label1",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)
            else:
                logitsLabel1 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])

            if 'Content' in name_prefix:
                logitsLabel0 = fc(x=fc7,
                                        output_size=len(self.config.datasetConfig.loadedLabel0Vec),
                                        scope="output_label0",
                                        parameter_update_device=self.networkInfo.device,
                                        weight_decay=weight_decay,
                                        initializer=self.initializer)
            else:
                logitsLabel0 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])
            
            outputLogits = [logitsLabel0, logitsLabel1]
            features=[conv1_2, conv2_2, conv3_4, conv4_4, conv5_4, fc6, fc7]

            if not reuse:
                self.varsTrain = [ii for ii in tf.trainable_variables() if name_prefix in ii.name]
                movingMeans=[ii for ii in tf.global_variables() if name_prefix in ii.name and 'moving_mean' in ii.name]
                movingVars=[ii for ii in tf.global_variables() if name_prefix in ii.name and 'moving_variance' in ii.name]
                varsSave=self.varsTrain+movingMeans+movingVars
                varsSave = reviseVarNames(varsSave,'-')
                self.saver = tf.train.Saver(max_to_keep=1, var_list=varsSave)
                print(name_prefix + "@" + self.networkInfo.device)
                PrintNetworkVars(name_prefix)
                print(print_separater)
            
            return outputLogits, features, fidFeature