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

from Networks.NetworkClass import NetworkIO

from Networks.Generators.Encoders.GeneralizeEncoder import GeneralizedEncoder as Encoder
from Networks.Generators.Decoders.GeneralizedDecoder import GeneralizedDecoder as Decoder
from Networks.Generators.Mixers.GeneralizedMixer import WNetMixer as Mixer

import os 
from Pipelines.ModuleBase import ModuleBase

eps = 1e-9
generator_dim = 64
print_separater="#########################################################"
class InstanceGeneratorIO(ModuleBase):
    def __init__(self, inputContents, inputStyles, displayStyles,
                 encodedContentFeatures, encodedStyleFeatures,
                 mixedFeatures, 
                 decodedFeatures, generated, groundtruths, 
                 constFeatures, categoryLogits):
        super().__init__()
        self.inputs.contents = inputContents
        self.inputs.styles = inputStyles
        self.inputs.displayStyles = displayStyles
        self.outputs.encodedContentFeatures=encodedContentFeatures
        self.outputs.encodedStyleFeatures=encodedStyleFeatures
        self.outputs.mixedFeatures=mixedFeatures
        self.outputs.decodedFeatures=decodedFeatures
        self.outputs.generated=generated
        
        self.outputs.constContent=constFeatures[0]
        self.outputs.constStyle=constFeatures[1]
        
        [self.outputs.categoryLogitContentOrg, 
         self.outputs.categoryLogitContentGenerated,
        self.outputs.categoryLogitStyleOrg, 
        self.outputs.categoryLogitStyleGenerated]=categoryLogits
        
        self.groundtruths.trueCharacter=groundtruths[0]
        self.groundtruths.onehotLabel0=groundtruths[1]
        self.groundtruths.onehotLabel1=groundtruths[2]
        self.groundtruths.denseLabel0=groundtruths[3]
        self.groundtruths.denseLabel1=groundtruths[4]
        
        

            
 
class GeneratorDataset(object):
    def __init__(self, config, dataIterator):
        self.contents = [dataIterator.output_tensor_list[1]]
        self.styles=list()
        for ii in range(config.datasetConfig.displayStyleNum):
            this_style_reference = tf.expand_dims(dataIterator.output_tensor_list[2][:,:,:,ii], axis=3)
            self.styles.append(this_style_reference)
        self.groundtruthCharacter=dataIterator.output_tensor_list[0]
        self.groundtruthLabels=[dataIterator.output_tensor_list[3],
                                dataIterator.output_tensor_list[4],
                                dataIterator.output_tensor_list[5],
                                dataIterator.output_tensor_list[6]]
        
        

class WNetGeneratorBase(object):
    def __init__(self, dataLoader, config, penalties):
        
        self.config=config
        self.penalties=penalties
        namePrefix = "Generator-"+self.config.generator.network
        self.initializer = 'XavierInit'
        self.IO=NetworkIO()
        
        
        
        # register the datasets for both training and validation
        self.trainData = GeneratorDataset(config=self.config, 
                                          dataIterator=dataLoader.train_iterator)
        
        self.validateData = GeneratorDataset(config=self.config, 
                                             dataIterator=dataLoader.validate_iterator)
        
        # register the content encoder
        self.contentEncoder = Encoder(config=self.config,
                                      scope=namePrefix+'/ContentEncoder',
                                      initializer = self.initializer,
                                      penalties=self.penalties)
    
        # register the style encoder
        self.styleEncoder = Encoder(config=self.config,
                                    scope=namePrefix+'/StyleEncoder',
                                    initializer = self.initializer,
                                    penalties=self.penalties)
        
        # register the feature mixer
        self.mixer = Mixer(config=self.config,
                           inputFromContentEncoder=self.contentEncoder.outputs,
                           inputFromStyleEncoder=self.styleEncoder.outputs,
                           initializer=self.initializer,
                           penalties=self.penalties,
                           scope=namePrefix+'/Mixer')

        # register the decoder
        self.decoder = Decoder(config=config,
                               inputFromContentEncoder=self.contentEncoder.outputs,
                               inputFromStyleEncoder=self.styleEncoder.outputs,
                               inputFromMixer=self.mixer.outputs,
                               initializer=self.initializer,
                               penalties=self.penalties,
                               scope=namePrefix+'/Decoder')
        
        
        # register the content encoder to calculate the const loss
        self.contentEncoderForConstLoss = Encoder(config=self.config,
                                                  scope=namePrefix+'/ContentEncoder',
                                                  initializer = self.initializer,
                                                  penalties=self.penalties)
        
        self.styleEncoderForConstLoss = Encoder(config=self.config,
                                                scope=namePrefix+'/StyleEncoder',
                                                initializer = self.initializer,
                                                penalties=self.penalties)
        

    def BuildGenerator(self, isTraining, validateOn='NA', saveEpochs=-1):
        if isTraining and validateOn=='NA':
            reuseLayer=False
            data=self.trainData
        else:
            reuseLayer=True
            if validateOn=='Trainset':
                data=self.trainData
            elif validateOn=='Validateset':
                data=self.validateData
                # data=self.validateData
            
                
        # Building the Content Prototype Encoder
        # if isTraining and validateOn=='NA':
        #     tmp_input=tf.constant(0, shape=data.contents[0].shape, dtype=tf.float32)
        # else:
        #     tmp_input=data.contents[0]
            
        self.contentEncoder.InitOutputLists()
        self.contentEncoder.BuildEncoder(inputImage=data.contents[0],
                                         is_training=isTraining, 
                                         reuse=reuseLayer,
                                         loadedCategoryLength=len(self.config.datasetConfig.loadedLabel0Vec),
                                         residual_connection_mode='Multi', 
                                         saveEpochs=saveEpochs)
        
        
        # Building the Style Reference Encoder
        self.styleEncoder.InitOutputLists()
        for ii in range(self.config.datasetConfig.inputStyleNum):
            if ii==0 and isTraining: 
                _thisReuseLayer=reuseLayer
            else: 
                _thisReuseLayer=True
            self.styleEncoder.BuildEncoder(inputImage=data.styles[ii],
                                           is_training=isTraining, 
                                           reuse=_thisReuseLayer,
                                           loadedCategoryLength=len(self.config.datasetConfig.loadedLabel1Vec),
                                           residual_connection_mode='Single',
                                           encoder_counter=ii, 
                                           saveEpochs=saveEpochs)
        self.styleEncoder=self.styleEncoder.ReorganizeOutputList(repeats=self.config.datasetConfig.inputStyleNum)
        
        
        
        # Building the mixer
        self.mixer.BuildMixer(is_training=isTraining, 
                              reuse=reuseLayer, 
                              saveEpochs=saveEpochs)
        
        # Building the decoder
        self.decoder.BuildDecoder(is_training=isTraining, reuse=reuseLayer, 
                                  saveEpochs=saveEpochs)
        # if not self.config.userInterface.resumeTrain and isTraining:
        #     input("Press Enter to continue...")
        
        
        # Building the const loss for the content
        self.contentEncoderForConstLoss.InitOutputLists()
        self.contentEncoderForConstLoss.BuildEncoder(inputImage=tf.tile(self.decoder.outputs.generated, 
                                                                         [1,1,1,
                                                                          int(self.contentEncoder.inputs.inputImg[0].shape[3])]),
                                                     is_training=isTraining, 
                                                     reuse=True,
                                                     loadedCategoryLength=len(self.config.datasetConfig.loadedLabel0Vec),
                                                     residual_connection_mode='Multi')
        
        # Building the const loss for the style
        self.styleEncoderForConstLoss.InitOutputLists()
        self.styleEncoderForConstLoss.BuildEncoder(inputImage=self.decoder.outputs.generated,
                                                   is_training=isTraining, 
                                                   reuse=True,
                                                   loadedCategoryLength=len(self.config.datasetConfig.loadedLabel1Vec),
                                                   residual_connection_mode='Single')
        
        
        # Register the IO
        thisIO=InstanceGeneratorIO(inputContents=self.contentEncoder.inputs.inputImg, 
                                   inputStyles=self.styleEncoder.inputs.inputImg, 
                                   displayStyles=data.styles,
                                   encodedContentFeatures=self.contentEncoderForConstLoss.outputs.encodedFinalOutputList[0].cnn, 
                                   encodedStyleFeatures=self.styleEncoderForConstLoss.outputs.encodedFinalOutputList[0].cnn, 
                                   mixedFeatures=self.mixer.outputs.fusedFeatures, 
                                   decodedFeatures=self.decoder.outputs.fullFeatureList, 
                                   generated=self.decoder.outputs.generated,
                                   groundtruths=[data.groundtruthCharacter]+data.groundtruthLabels,
                                   constFeatures=[self.contentEncoder.outputs.encodedFinalOutputList[0].cnn,
                                                  [ii[0].cnn for ii in self.styleEncoder.outputs.encodedFinalOutputList]],
                                   categoryLogits=[self.contentEncoder.outputs.category, self.contentEncoderForConstLoss.outputs.category,
                                                   self.styleEncoder.outputs.category, self.styleEncoderForConstLoss.outputs.category]) 
        
        
        return thisIO, \
            [self.contentEncoder.saver, self.styleEncoder.saver, self.mixer.saver, self.decoder.saver], \
            [self.contentEncoder.varsTrain,  self.styleEncoder.varsTrain, self.mixer.varsTrain, self.decoder.varsTrain], \
                [os.path.join(self.config.generator.path,'ContentEncoder'), 
                 os.path.join(self.config.generator.path,'StyleEncoder'), 
                 os.path.join(self.config.generator.path,'Mixer'), 
                 os.path.join(self.config.generator.path,'Decoder')]
        
