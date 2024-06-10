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

from Networks.NetworkClass import NetworkBase
from Utilities.utils import PrintNetworkVars
from Utilities.VitTools import VitImplementation as vit

# from Utilities.Blocks import EncodingBasicBlock as EncodingBasicBlock
# from Utilities.Blocks import DecodingBasicBlock as DecodingBasicBlock

from Utilities.Blocks import EncodingBottleneckBlock as EncodingBottleneckBlock
from Utilities.Blocks import DecodingBottleneckBlock as DecodingBottleneckBlock
from Utilities.Blocks import EncodingVisionTransformerBlock as EncodingVisionTransformerBlock
from Utilities.Blocks import DecodingVisionTransformerBlock as DecodingVisionTransformerBlock
from Utilities.Blocks import FusingStyleFeatures

from prettytable import PrettyTable
from Utilities.utils import SplitName
from Utilities.Blocks import BlockFeature

eps = 1e-9
cnnDim=64
vitDim=96
from Utilities.Blocks import patchSize

def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)
BlockEncDict={'Cv': EncodingBottleneckBlock,
           'Cbb': EncodingBottleneckBlock,
           'Cbn': EncodingBottleneckBlock,
           'Vit': EncodingVisionTransformerBlock}

BlockDecDict={'Cv': DecodingBottleneckBlock,
           'Cbb': DecodingBottleneckBlock,
           'Cbn': DecodingBottleneckBlock,
           'Vit': DecodingVisionTransformerBlock}


StyleFusingDict={'Max': tf.reduce_max,
                'Min': tf.reduce_min,
                'Avg': tf.reduce_mean}
# residualAtLayer=3
# residualBlockNum=5

print_separater="#########################################################"



class WNetMixer(NetworkBase):
    def __init__(self, inputFromContentEncoder, inputFromStyleEncoder, 
                 config, scope, initializer, penalties):
        
        super().__init__()
        
        
        self.config = config
        _, self.styleFusion, fusionContentStyle  = SplitName(config.generator.mixer)[:3]
        if 'Res' in fusionContentStyle: # Residual
            self.fusionContentStyle='Res'
        elif 'Dns' in fusionContentStyle: # Dense
            self.fusionContentStyle='Dns'
        elif 'Smp' in fusionContentStyle:
            self.fusionContentStyle='Smp' # Simple Connection
        
        if self.fusionContentStyle=='Dns' or self.fusionContentStyle=='Res':
            residualBlockNum, residualAtLayer = fusionContentStyle[len(self.fusionContentStyle):].split('@')
            self.residualBlockNum=int(residualBlockNum)
            self.residualAtLayer=int(residualAtLayer)
            
        
        self.scope=scope
        self.initializer = initializer
        self.penalties = penalties
        
        self.inputs.update({'fromContentEncoder': inputFromContentEncoder})
        self.inputs.update({'fromStyleEncoder': inputFromStyleEncoder})
        self.outputs.update({'fusedFeatures': None})
        self.outputs.update({'encodedStyleFinalOutput': None})
        self.outputs.update({'styleShortcutBatchDiff': None})
        self.outputs.update({'styleResidualBatchDiff': None})
        
        self.architectureEncoderList = SplitName(self.config.generator.encoder)[1:]
        self.architectureDecoderList = SplitName(self.config.generator.decoder)[1:]
        
        if self.architectureEncoderList[-1]==self.architectureDecoderList[0]:
            self.lastFusing=self.architectureEncoderList[-1]
        elif 'Vit' in self.architectureEncoderList[-1]:
            self.lastFusing =self.architectureEncoderList[-1]
        elif 'Vit' in self.architectureDecoderList[0]:
            self.lastFusing =self.architectureDecoderList[0]
        else:
            self.lastFusing=self.architectureDecoderList[0]
        
        return
    
    def FuseStyleFeature(self, is_training):
        
        # fusing the final encoded      
        
        fusedFinalStyle=\
            FusingStyleFeatures(repeatNum=self.config.datasetConfig.inputStyleNum, 
                                fusingList=self.inputs.fromStyleEncoder.encodedFinalOutputList, 
                                fusionMethod=self.styleFusion, 
                                needAct=False, 
                                architecture=self.lastFusing, 
                                is_training=is_training, 
                                scope='StyleFeatureFuse-FinalLayer', 
                                weightDecay=self.thisWeightDecay, 
                                initializer=self.initializer, 
                                device=self.config.generator.device, 
                                outputMark='toNext')
        
        
        # fusing the full features
        fusedFullStyleFeatureList=list()
        for jj in range(len(self.inputs.fromStyleEncoder.fullFeatureList[0])): 
            
            thisEvaluateList=[[self.inputs.fromStyleEncoder.fullFeatureList[kk][jj]] for kk in range(len(self.inputs.fromStyleEncoder.fullFeatureList))]
                                          
            thisFusedFinalStyle=\
                FusingStyleFeatures(repeatNum=self.config.datasetConfig.inputStyleNum, 
                                    fusingList=thisEvaluateList, 
                                    fusionMethod=self.styleFusion, 
                                    needAct=True, 
                                    architecture=self.architectureEncoderList[jj], 
                                    is_training=is_training, 
                                    scope='StyleFeatureFuse-Layer%d' % jj, 
                                    weightDecay=self.thisWeightDecay, 
                                    initializer=self.initializer, 
                                    device=self.config.generator.device, 
                                    outputMark='toDecoder')
            fusedFullStyleFeatureList.append(thisFusedFinalStyle)
            
        return fusedFullStyleFeatureList, fusedFinalStyle
    
    
    def FuseContentAndStyleFeatures(self, fusedStyles, fusedFinalStyle):
        
        fusedContentStyle=list()
        for ii in range(len(fusedStyles)):
            thisStyle = fusedStyles[ii]
            thisContent = self.inputs.fromContentEncoder.fullFeatureList[ii] 
            thisFusedCNN = tf.concat([thisContent.cnn, thisStyle.cnn], axis=-1)
            thisFusedVit = tf.concat([thisContent.vit, thisStyle.vit], axis=-1)
            thisFused=BlockFeature(cnn=thisFusedCNN, vit=thisFusedVit)
            fusedContentStyle.append(thisFused)

        fusedFinalContentStyleCNN = tf.concat([self.inputs.fromContentEncoder.encodedFinalOutputList[0].cnn, fusedFinalStyle.cnn], axis=-1)
        fusedFinalContentStyleVit = tf.concat([self.inputs.fromContentEncoder.encodedFinalOutputList[0].vit, fusedFinalStyle.vit], axis=-1)
        fusedFinalContentStyle=BlockFeature(cnn=fusedFinalContentStyleCNN, vit=fusedFinalContentStyleVit)
        
        return  fusedContentStyle, fusedFinalContentStyle
    
    
    def BuildMixer(self,
                   is_training,
                   reuse = False, saveEpochs=-1):
        
            
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.config.generator.device):
                with tf.variable_scope(self.scope):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                        self.thisWeightDecay = -1
                    else:
                        self.thisWeightDecay = self.penalties.generator_weight_decay_penalty
        
        
        
                    fusedStyles, fusedFinalStyles = self.FuseStyleFeature(is_training=is_training)
                    fusedContentStyle, fusedFinalContentStyle = self.FuseContentAndStyleFeatures(fusedStyles, fusedFinalStyles)
                        
                        
                    # Fused feature process
                    fusedFeatures=list()
                    if self.fusionContentStyle is not 'Smp': # not simple mixer
                        for ii in range(len(fusedContentStyle)):
                            thisFeatureCNN = fusedContentStyle[ii].cnn
                            thisFeatureVit = fusedContentStyle[ii].vit
                            if ii <self.residualAtLayer:
                                thisResidualNum = max(self.residualBlockNum-ii*2,1)
                                
                                channelListCNN = [int(ii) for ii in np.linspace(int(thisFeatureCNN.shape[-1]),
                                                                                int(thisFeatureCNN.shape[-1])//2,
                                                                                thisResidualNum+1).tolist()]
                                channelListVit = [int(ii) for ii in np.linspace(int(thisFeatureVit.shape[-1]),
                                                                                int(thisFeatureVit.shape[-1])//2,
                                                                                thisResidualNum+1).tolist()]
                                
                                inputFeature=fusedContentStyle[ii]
                                with tf.variable_scope('FeatureProcessBlock-Layer%d' % ii):
                                    for jj in range(thisResidualNum):
                                        thisBlock = FindKeys(BlockEncDict, self.architectureEncoderList[ii])[0]
                                        thisCnnHW=int(thisFeatureCNN.shape[1])
                                        thisVitDim = int(thisFeatureVit.shape[1])
                                        fusedContentStyle[ii].cnn=lrelu(fusedContentStyle[ii].cnn)
                                        resultFeature = thisBlock(input=inputFeature,  
                                                                  blockCount=jj+1, 
                                                                  is_training=is_training,  
                                                                  dims={'HW': thisCnnHW,  'MapC': channelListCNN[jj+1],
                                                                        'VitC': channelListVit[jj+1], 'VitDim': thisVitDim}, 
                                                                  config={'option': self.architectureEncoderList[ii]},
                                                                  weightDecay=self.thisWeightDecay, 
                                                                  initializer=self.initializer,  
                                                                  device=self.config.generator.device)
                                        inputFeature=resultFeature.toNext
                                    resultFeature = resultFeature.toDecoder
                            else:
                                resultFeature = fusedContentStyle[ii]
                            fusedFeatures.append(resultFeature)
                    else:
                        fusedFeatures = fusedContentStyle
                    
                    self.outputs.fusedFeatures=fusedFeatures
                    
                    # process the last:
                    # input 2x2xDIM (need upsample) / 1xDIM (no upsample needed), output 4x4xdim*16 / 1xdim*16
                    # here, cnn needs to be upsampled, but vit needs not
                    thisBlock = FindKeys(BlockDecDict, self.lastFusing)[0]
                    downVitDim = (self.config.datasetConfig.imgWidth // patchSize )**2
                    final = \
                            thisBlock(x=fusedFinalContentStyle,
                                      dims={'MapC': cnnDim*16, 'VitC': vitDim*16,
                                            'HW': self.config.datasetConfig.imgWidth//16,'VitDim': downVitDim//256}, 
                                    config={'option': self.lastFusing},
                                    blockCount=0, 
                                    device=self.config.generator.device,
                                    weightDecay=self.penalties.generator_weight_decay_penalty,
                                    initializer=self.initializer,
                                    isTraining=is_training)
                    self.outputs.encodedFinalOutput = final
                    # self.outputs.fullFeatureList.append(result0)
        
        if is_training and not reuse:
            self.varsTrain = [ii for ii in tf.trainable_variables() if self.scope in ii.name]
            movingMeans=[ii for ii in tf.global_variables() if self.scope in ii.name and 'moving_mean' in ii.name]
            movingVars=[ii for ii in tf.global_variables() if self.scope in ii.name and 'moving_variance' in ii.name]
            varsSave=self.varsTrain+movingMeans+movingVars
            self.saver=None
            if not len(varsSave)==0:
                self.saver = tf.train.Saver(max_to_keep=saveEpochs, var_list=varsSave)                
            print(self.scope + "@" + self.config.generator.device)
            PrintNetworkVars(self.scope)
            
            table = PrettyTable(['Layer', 'FusedStyleCNN','FusedStyleVit', 'FusedContentStyleCNN','FusedContentStyleVit','FusedFeaturesCNN','FusedFeaturesVit'])
            table.add_row(['0']+fusedStyles[0].ProcessOutputToList()+fusedContentStyle[0].ProcessOutputToList()+fusedFeatures[0].ProcessOutputToList())
            table.add_row(['1']+fusedStyles[1].ProcessOutputToList()+fusedContentStyle[1].ProcessOutputToList()+fusedFeatures[1].ProcessOutputToList())
            table.add_row(['2']+fusedStyles[2].ProcessOutputToList()+fusedContentStyle[2].ProcessOutputToList()+fusedFeatures[2].ProcessOutputToList())
            table.add_row(['3']+fusedStyles[3].ProcessOutputToList()+fusedContentStyle[3].ProcessOutputToList()+fusedFeatures[3].ProcessOutputToList())
            
            print(table)     
            
            table = PrettyTable(['The Last Layer', 'CNN','ViT'])
            table.add_row(['LastFusion']+final.ProcessOutputToList())
            print(table)     
            
            print(print_separater)
        return
        
            