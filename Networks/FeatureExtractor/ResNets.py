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
import re
import math

from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock
from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
from Utilities.utils import PrintNetworkVars
# from Networks.FeatureExtractor.Framework import FeatureExtractor


eps = 1e-9
output_filters = 64

filter_list=[64,128,256,512]
        
res18_block_list=[2,2,2,2]
res34_block_list=[3,4,6,3]
res50_block_list=[3,4,6,3]
res101_block_list=[3,4,23,3]
res152_block_list=[3,8,36,3]

resblock_dict={'resnet18':res18_block_list,
                'resnet34':res34_block_list,
                'resnet50':res50_block_list,
                'resnet101':res101_block_list,
                'resnet152':res152_block_list}

print_separater="#########################################################"


def reviseVarNames(var_input,deleteFrom):
        var_output = {}
        for ii in var_input:
                prefix_pos = ii.name.find(deleteFrom)
                renamed = ii.name[prefix_pos + 1:]
                parafix_pos = renamed.find(':')
                renamed = renamed[0:parafix_pos]
                var_output.update({renamed: ii})
        return var_output

        
class ResNets(object):
    def __init__(self,
                 config, namePrefix, initializer, penalties, networkInfo):
        
        # super().__init__()
        self.config = config
        self.namePrefix = namePrefix
        self.initializer = initializer
        self.penalties = penalties
        
        self.networkInfo = networkInfo

        return


    def NetworkImplementation(self, inputImg, 
                              weight_decay=-1,
                              is_training=False,
                              name_prefix='None',
                              reuse=False):
    

        name_prefix_correction_1 = name_prefix.find("resnet")
        key=re.findall(r"\d+", name_prefix)
        if len(key):
                name_prefix_correction_2 = name_prefix.find(re.findall(r"\d+", name_prefix)[0])
                name_prefix_correction = name_prefix[name_prefix_correction_1:name_prefix_correction_2+len(re.findall(r"\d+", name_prefix)[0])]
                res_block_list=resblock_dict[name_prefix_correction]
        else:
                res_block_list=resblock_dict[name_prefix]

        if '18' in name_prefix or '34' in name_prefix:
                resblock=self.basic_block
        elif '50' in name_prefix or '101' in name_prefix or '152'  in name_prefix:
                resblock=self.bottleneck_block
        
        if is_training:
                print(print_separater)
                print("Training on " + name_prefix)
                print(print_separater)
        if is_training:
            keep_prob=0.5
        else:
            keep_prob=1.0

        with tf.variable_scope(name_prefix):
            if reuse:
                tf.get_variable_scope().reuse_variables()
        
            ## block 1
            conv1 = relu(batch_norm(x=conv2d(x=inputImg, output_filters=64,
                                            kh=7, kw=7,
                                            sh=2, sw=2,             
                                            padding='SAME',
                                            parameter_update_device=self.networkInfo.device,
                                            weight_decay=weight_decay,
                                            initializer=self.initializer,
                                            scope='conv1'),
                                    is_training=is_training,
                                    scope='bn1',
                                    parameter_update_device=self.networkInfo.device))
            
            ## block 2
            pool1 =tf.nn.max_pool(value=conv1,
                                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool1')
            
            res2 = resblock(num_blocks=res_block_list[0], num_features=filter_list[0], 
                            input_feature = pool1, weight_decay=weight_decay, is_training=is_training, 
                            name_prefix='resblock2', downsample=False)

            ## block 3
            res3 = resblock(num_blocks=res_block_list[1], num_features=filter_list[1], 
                            input_feature = res2, weight_decay=weight_decay, is_training=is_training, 
                            name_prefix='resblock3', downsample=True)

            ## block 4
            res4 = resblock(num_blocks=res_block_list[2], num_features=filter_list[2], 
                            input_feature = res3, weight_decay=weight_decay, is_training=is_training, 
                            name_prefix='resblock4', downsample=True)

            ## block 5
            res5 = resblock(num_blocks=res_block_list[3], num_features=filter_list[3], 
                            input_feature = res4, weight_decay=weight_decay, is_training=is_training, 
                            name_prefix='resblock5', downsample=True)

            fidFeature=res5

            ## block 6
            pool6 = tf.nn.avg_pool2d(value=res5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool6')
            fc6 = tf.reshape(pool6, [self.config.trainParams.batchSize, -1])
            
            
            if 'Style' in name_prefix:
                logitsLabel1 = fc(x=fc6,
                                output_size=len(self.config.datasetConfig.loadedLabel1Vec),
                                scope="output_label1",
                                parameter_update_device=self.networkInfo.device,
                                weight_decay=weight_decay,
                                initializer=self.initializer)
                
            else:
                logitsLabel1 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])

            if 'Content' in name_prefix:
                logitsLabel0 = fc(x=fc6,
                                output_size=len(self.config.datasetConfig.loadedLabel0Vec),
                                scope="output_label0",
                                parameter_update_device=self.networkInfo.device,
                                weight_decay=weight_decay,
                                initializer=self.initializer)
            else:
                logitsLabel0 = tf.constant(value=-1, dtype=tf.float32, shape=[self.config.trainParams.batchSize,1])
            
            outputLogits=[logitsLabel0,logitsLabel1]
            features=[conv1, res2, res3, res4, res5, fc6]

    
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
    
    
    
    def basic_block(self,
                    num_blocks, num_features, input_feature, weight_decay, is_training, name_prefix, downsample):
        identity = input_feature
        feature = input_feature
        
        for ii in range(num_blocks):
        
            if downsample and ii==0:
                feature = conv2d(x=feature, output_filters=num_features, kh=3, kw=3, sh=2, sw=2, 
                                padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                                initializer=self.initializer,  scope=name_prefix+'_conv%d_1' % (ii+1))
            else:
                feature = conv2d(x=feature, output_filters=num_features, kh=3, kw=3, sh=1, sw=1, 
                                padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                                initializer=self.initializer,  scope=name_prefix+'_conv%d_1'% (ii+1))
            feature = batch_norm(feature, is_training=is_training,scope=name_prefix+'_bn%d_1'% (ii+1), parameter_update_device=self.networkInfo.device)
            feature = relu(feature)

            feature = conv2d(x=feature, output_filters=num_features, kh=3, kw=3, sh=1, sw=1, 
                             padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                             initializer=self.initializer,  scope=name_prefix+'_conv%d_2' % (ii+1))
            feature = batch_norm(feature, is_training=is_training,scope=name_prefix+'_bn%d_2'% (ii+1), parameter_update_device=self.networkInfo.device)

            if not ii==num_blocks-1:
                feature = relu(feature)

        if downsample:
            identity = conv2d( x=identity, output_filters=num_features, kh=1, kw=1, sh=2, sw=2, 
                        padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                        initializer=self.initializer,  scope=name_prefix+'_conv_downsample')
            identity=batch_norm(identity, is_training=is_training,scope=name_prefix+'_bn_downsample', parameter_update_device=self.networkInfo.device)

        out = identity + feature
        out = relu(out)
        return out
    
    
    def bottleneck_block(self,
                         num_blocks, num_features, input_feature, weight_decay, is_training, name_prefix, downsample):
        identity = input_feature
        feature = input_feature
        channel_expansion=4

        for ii in range(num_blocks):
            if downsample and ii==0:
                feature = conv2d( x=feature, output_filters=num_features, kh=1, kw=1, sh=2, sw=2, 
                                padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                                initializer=self.initializer,  scope=name_prefix+'_conv%d_1' % (ii+1))
            else:
                feature = conv2d( x=feature, output_filters=num_features, kh=1, kw=1, sh=1, sw=1, 
                                padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                                initializer=self.initializer, scope=name_prefix+'_conv%d_1'% (ii+1))

            feature = batch_norm(feature, is_training=is_training,scope=name_prefix+'_bn%d_1'% (ii+1), parameter_update_device=self.networkInfo.device)
            feature = relu(feature)

            
            
            feature = conv2d( x=feature, output_filters=num_features, kh=3, kw=3, sh=1, sw=1, 
                                padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                                initializer=self.initializer, scope=name_prefix+'_conv%d_2'% (ii+1))

            feature = batch_norm(feature, is_training=is_training,scope=name_prefix+'_bn%d_2'% (ii+1), parameter_update_device=self.networkInfo.device)
            feature = relu(feature)

            
            
            feature = conv2d( x=feature, output_filters=num_features * channel_expansion, kh=1, kw=1, sh=1, sw=1, 
                            padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                            initializer=self.initializer,  scope=name_prefix+'_conv%d_3' % (ii+1))
            feature = batch_norm(feature, is_training=is_training,scope=name_prefix+'_bn%d_3'% (ii+1), parameter_update_device=self.networkInfo.device)

            if not ii==num_blocks-1:
                feature = relu(feature)

        if downsample:
            identity = conv2d( x=identity, output_filters=num_features * channel_expansion, kh=1, kw=1, sh=2, sw=2, 
                        padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                        initializer=self.initializer,  scope=name_prefix+'_conv_downsample')
            identity=batch_norm(identity, is_training=is_training,scope=name_prefix+'_bn_downsample', parameter_update_device=self.networkInfo.device)
        else:
            identity = conv2d( x=identity, output_filters=num_features * channel_expansion, kh=1, kw=1, sh=1, sw=1, 
                        padding='SAME', parameter_update_device=self.networkInfo.device, weight_decay=weight_decay,
                        initializer=self.initializer,  scope=name_prefix+'_conv_downsample')
            identity=batch_norm(identity, is_training=is_training,scope=name_prefix+'_bn_downsample', parameter_update_device=self.networkInfo.device)

        out = identity + feature
        out = relu(out)
        return out