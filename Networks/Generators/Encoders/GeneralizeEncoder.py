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

from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock, maxPool
from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock

from Networks.NetworkClass import EncoderBase
from Utilities.utils import PrintNetworkVars
from Utilities.Blocks import EncodingBasicBlock as BasicBlock
from Utilities.Blocks import EncodingBottleneckBlock as BottleneckBlock
from Utilities.Blocks import EncodingVisionTransformerBlock as VisionTransformerBlock
from prettytable import PrettyTable
from Utilities.utils import SplitName
from Utilities.Blocks import BlockFeature

eps = 1e-9
cnnDim = 64
vitDim = 96
from Utilities.Blocks import patchSize
BlockDict={'Cv': BasicBlock,
           'Cbb': BasicBlock,
           'Cbn': BottleneckBlock,
           'Vit': VisionTransformerBlock}
def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)
# numHeads=[-1, 3,6,12,24]
# depths=[-1, 2,2,2,2]
numHeads=[-1, -1,-1,8,-1]
depths=[-1, -1,-1,12,-1]

print_separater="#########################################################"


class GeneralizedEncoder(EncoderBase):
    def __init__(self,
                 config, scope, initializer, penalties):
        
        super().__init__()
        self.config = config
        self.scope=scope
        self.initializer = initializer
        self.penalties = penalties
        
        return
    

    def BuildEncoder(self, inputImage,
                     is_training,residual_connection_mode, loadedCategoryLength, 
                     reuse = False,
                     encoder_counter=0, saveEpochs=-1):
        architectureList = SplitName(self.config.generator.encoder)[1:]
        if encoder_counter==0:
            self.inputs.update({'inputImg': list()})
        self.inputs.inputImg.append(inputImage)
        
        if is_training:
            keep_prob=0.5
        else:
            keep_prob=1.0
        
        if encoder_counter==0:
            this_reuse=reuse
            this_weight_decay = self.penalties.generator_weight_decay_penalty
        else:
            this_reuse=True
            this_weight_decay = -1

        if is_training and not reuse:
            print(print_separater)
            print("Build the Generator: " + self.config.generator.name)

        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.config.generator.device):
                with tf.variable_scope(self.scope):
                    if this_reuse:
                        tf.get_variable_scope().reuse_variables()

                    # Block 0 as STEM
                    downImput = BlockFeature(cnn=self.inputs.inputImg[encoder_counter])
                    result0=\
                        BottleneckBlock(input=downImput,  
                                                       blockCount=0,  
                                                       is_training=is_training,  
                                                       dims={'HW':self.config.datasetConfig.imgWidth, 
                                                             'MapC': cnnDim//2,
                                                             'VitC': -1,
                                                             'VitDim': -1}, 
                                                       config={'option': 'Cbn'},
                                                       weightDecay=this_weight_decay, initializer=self.initializer,  
                                                       device=self.config.generator.device)
                    # self.outputs.fullFeatureList.append(result1.toDecoder)
                    
                    
                    # Block 1
                    downVitDim = (self.config.datasetConfig.imgWidth // patchSize )**2
                    thisBlock = FindKeys(BlockDict, architectureList[0])[0]
                    result1=\
                        thisBlock(input=result0.toNext,  
                                  blockCount=1,  
                                                       is_training=is_training,  
                                                       dims={'HW':self.config.datasetConfig.imgWidth//2, 
                                                             'MapC': cnnDim,
                                                             'VitC': vitDim,
                                                             'VitDim': downVitDim//4}, 
                                                       config={'option': architectureList[0]},
                                                       weightDecay=this_weight_decay, initializer=self.initializer,  
                                                       device=self.config.generator.device)
                    self.outputs.fullFeatureList.append(result1.toDecoder)
                    
                    
                    # Block 2
                    thisBlock = FindKeys(BlockDict, architectureList[1])[0]
                    result2 = thisBlock(input=result1.toNext,  
                                        blockCount=2, 
                                        is_training=is_training,  
                                        dims={'HW': self.config.datasetConfig.imgWidth//4, 
                                                             'MapC': cnnDim*2,
                                                             'VitC': vitDim*2,
                                                             'VitDim': downVitDim//16}, 
                                        config={'option': architectureList[1],
                                                               'numViT': depths[1],
                                                               'numHead': numHeads[1]},
                                        weightDecay=this_weight_decay, 
                                        initializer=self.initializer,  
                                        device=self.config.generator.device)
                    self.outputs.fullFeatureList.append(result2.toDecoder)
                    
                        
                    # Block 3
                    thisBlock = FindKeys(BlockDict, architectureList[2])[0]
                    result3=thisBlock(input=result2.toNext,  
                                                       blockCount=3,  
                                                       is_training=is_training,  
                                                       dims={'HW':self.config.datasetConfig.imgWidth//8, 
                                                             'MapC': cnnDim*4,
                                                             'VitC': vitDim*4,
                                                             'VitDim': downVitDim//64}, 
                                                       config={'option': architectureList[2],
                                                               'numViT': depths[2],
                                                               'numHead': numHeads[2]},
                                                       weightDecay=this_weight_decay, 
                                                       initializer=self.initializer,  
                                                       device=self.config.generator.device)
                    self.outputs.fullFeatureList.append(result3.toDecoder)
                    
                        
                    # Block 4
                    thisBlock = FindKeys(BlockDict, architectureList[3])[0]
                    result4 = thisBlock(input=result3.toNext,  
                                        blockCount=4,  
                                        is_training=is_training,  
                                        dims={'HW':self.config.datasetConfig.imgWidth//16, 
                                              'MapC': cnnDim*8,
                                              'VitC': vitDim*8,
                                              'VitDim': downVitDim//256}, 
                                                       config={'option': architectureList[3],
                                                               'numViT': depths[3],
                                                               'numHead': numHeads[3]},
                                                       weightDecay=this_weight_decay, 
                                                       initializer=self.initializer,  
                                                       device=self.config.generator.device)
                    self.outputs.fullFeatureList.append(result4.toDecoder)
                    self.outputs.encodedFinalOutputList.append(result4.toNext)
                    
                        
                    # # Block 4
                    # thisBlock = FindKeys(BlockDict, architectureList[4])[0]
                    # result4 =thisBlock(input=result3.toNext,  blockCount=4,  
                    #                    is_training=is_training,  
                    #                    dims={'HW':self.config.datasetConfig.imgWidth//32, 
                    #                          'MapC': cnnDim*16,
                    #                          'VitC': vitDim*16,
                    #                          'VitDim': downVitDim//256}, 
                    #                    config={'option': architectureList[4],
                    #                            'numViT': depths[4],
                    #                            'numHead': numHeads[4]},
                    #                    weightDecay=this_weight_decay, 
                    #                    initializer=self.initializer,  
                    #                    device=self.config.generator.device,  
                    #                    scope=self.scope, 
                    #                    option=architectureList[4])
                    # self.outputs.fullFeatureList.append(result4.toDecoder)
                    # self.outputs.encodedFinalOutputList.append(result4.toNext)
                    
                    
                    # Block FC for category
                    fc1 = tf.reshape(result3.toNext.cnn, [self.config.trainParams.batchSize, -1])
                    fc1 = tf.nn.dropout(relu(fc(x=fc1,
                                  output_size=4096,
                                  scope="BlockFC1",
                                  parameter_update_device=self.config.generator.device,
                                  weight_decay=this_weight_decay,
                                  initializer=self.initializer)),
                                        keep_prob=keep_prob)
                    
                    category = fc(x=fc1,
                                  output_size=loadedCategoryLength,
                                  scope="FinalCategory",
                                  parameter_update_device=self.config.generator.device,
                                  weight_decay=this_weight_decay,
                                  initializer=self.initializer)
                    
                    self.outputs.category.append(category)
                    
            if is_training and not reuse:
                self.varsTrain = [ii for ii in tf.trainable_variables() if self.scope in ii.name]
                movingMeans=[ii for ii in tf.global_variables() if self.scope in ii.name and 'moving_mean' in ii.name]
                movingVars=[ii for ii in tf.global_variables() if self.scope in ii.name and 'moving_variance' in ii.name]
                varsSave=self.varsTrain+movingMeans+movingVars
                self.saver = tf.train.Saver(max_to_keep=saveEpochs, var_list=varsSave)
                
                print(self.scope + "@" + self.config.generator.device)
                PrintNetworkVars(self.scope, exceptions=["BlockFC1", "FinalCategory"])
                
                table = PrettyTable(['Layer', 'ToNextCNN','ToNextVit','ToDecoderCNN','ToDecoderVit'])
                table.add_row(['0-STEM-'+architectureList[0]]+ result0.toNext.ProcessOutputToList() + result0.toDecoder.ProcessOutputToList())
                table.add_row(['1-'+architectureList[0]]+ result1.toNext.ProcessOutputToList() + result1.toDecoder.ProcessOutputToList())
                table.add_row(['2-'+architectureList[1]]+ result2.toNext.ProcessOutputToList() + result2.toDecoder.ProcessOutputToList())
                table.add_row(['3-'+architectureList[2]]+ result3.toNext.ProcessOutputToList() + result3.toDecoder.ProcessOutputToList())
                table.add_row(['4-'+architectureList[3]]+ result4.toNext.ProcessOutputToList() + result4.toDecoder.ProcessOutputToList())
                # table.add_row(['4-'+architectureList[4]]+ result4.toNext.ProcessOutputToList() + result4.toDecoder.ProcessOutputToList())
                print(table)     
                print(print_separater)
                    
            return

    