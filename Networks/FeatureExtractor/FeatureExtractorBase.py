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

from Networks.NetworkClass import NetworkIO

from Networks.FeatureExtractor.VGGs import VGGs
from Networks.FeatureExtractor.ResNets import ResNets
from Pipelines.ModuleBase import ModuleBase


NetworkSelections = {'vgg16net': VGGs,
                     'vgg19net': VGGs,
                     'resnet18': ResNets,
                     'resnet34': ResNets,
                     'resnet50': ResNets,
                     'resnet101': ResNets,
                     'resnet152': ResNets}


eps = 1e-9
output_filters = 64
print_separater="#########################################################"



class InstanceFeatureExtractorIO(ModuleBase):
    def __init__(self):
        super().__init__()

        
        self.inputs.update({"imgReal": None})
        self.inputs.update({'imgFake': None})
        self.outputs.update({'realContentLogits': list()})
        self.outputs.update({'fakeContentLogits': list()})
        self.outputs.update({'realStyleLogits': list()})
        self.outputs.update({'fakeStyleLogits': list()})
        
        self.outputs.update({'realContentFidFeature': list()})
        self.outputs.update({'fakeContentFidFeature': list()})
        self.outputs.update({'realStyleFidFeature': list()})
        self.outputs.update({'fakeStyleFidFeature': list()})
        
        
        
        self.outputs.update({'realContentFeatures': list()})
        self.outputs.update({'fakeContentFeatures': list()})
        self.outputs.update({'realStyleFeatures': list()})
        self.outputs.update({'fakeStyleFeatures': list()})
                
        # self.savers.update({'contentExtractorSavers': list()})
        # self.savers.update({'styleExtractorSavers': list()})
        
        
        
        

class FeatureExtractor():
    def __init__(self,
                 config, penalties, dataLoader):
        
        
        self.config = config
        self.initializer = 'XavierInit'
        self.namePrefix = "FeatureExtractor"
        self.penalties = penalties
        self.dataLoader = dataLoader
        self.IO=NetworkIO()
        
            
        if 'extractorContent' in self.config:
            self.contentExtractors = list()
            for _model in self.config.extractorContent:
                _currentInit = NetworkSelections[_model.name]
                self.contentExtractors.append(_currentInit(config=self.config,
                                                           namePrefix=self.namePrefix+'VGG16', 
                                                           initializer=self.initializer, 
                                                           penalties=self.penalties, 
                                                           networkInfo=_model))
        
        if 'extractorStyle' in self.config:
            self.styleExtractors=list()
            for _model in self.config.extractorStyle:
                _currentInit = NetworkSelections[_model.name]
                self.styleExtractors.append(_currentInit(config=self.config,
                                                        namePrefix=self.namePrefix+'VGG16', 
                                                        initializer=self.initializer, 
                                                        penalties=self.penalties, 
                                                        networkInfo=_model))
        
        return
    
    def BuildFeatureExtractor(self, generatorIO, validateOn, reuse, isTrain=False):
        
        thisIO = InstanceFeatureExtractorIO()
        
        if isTrain:
            #thisGeneratorIO=generatorIO.train
            dataItr = self.dataLoader.train_iterator
        elif validateOn=='Trainset':
            #thisGeneratorIO=generatorIO.testOnTrain
            dataItr = self.dataLoader.train_iterator
        elif validateOn=='Validateset':
            #thisGeneratorIO=generatorIO.testOnValidate
            dataItr = self.dataLoader.validate_iterator
        thisIO.inputs.imgReal=generatorIO.groundtruths.trueCharacter
        thisIO.inputs.imgFake=generatorIO.outputs.generated
        thisIO.groundtruths.onehotLabel0=dataItr.output_tensor_list[3]
        thisIO.groundtruths.denseLabel0=dataItr.output_tensor_list[5]
        thisIO.groundtruths.onehotLabel1=dataItr.output_tensor_list[4]
        thisIO.groundtruths.denseLabel1=dataItr.output_tensor_list[6]
         
        
        contentExtractorSavers=[]
        styleExtractorSavers=[]
        contentExtractorPaths=[]
        styleExtractorPaths=[]
        #if self.config.extractorContent:
        if 'extractorContent' in self.config:
            for _model in self.contentExtractors:
                _realContentLogits, _realContentFeatures, _realFidContentFeature = \
                    _model.NetworkImplementation(inputImg=thisIO.inputs.imgReal, 
                                                 name_prefix=self.namePrefix+'/Content-'+_model.networkInfo.name,
                                                 is_training=False, reuse=reuse)
                thisIO.outputs.realContentLogits.append(_realContentLogits[0])
                thisIO.outputs.realContentFeatures.append(_realContentFeatures)
                thisIO.outputs.realContentFidFeature.append(_realFidContentFeature)
                if not reuse: 
                    contentExtractorSavers.append(_model.saver)
                    contentExtractorPaths.append(_model.networkInfo.path)
                
                _fakeContentLogits, _fakeContentFeatures, _fakeFidContentFeature = \
                    _model.NetworkImplementation(inputImg=thisIO.inputs.imgFake, 
                                                 name_prefix=self.namePrefix+'/Content-'+_model.networkInfo.name,
                                                 is_training=False, reuse=True)
                thisIO.outputs.fakeContentLogits.append(_fakeContentLogits[0])
                thisIO.outputs.fakeContentFeatures.append(_fakeContentFeatures)
                thisIO.outputs.fakeContentFidFeature.append(_realFidContentFeature)
                    
        #if self.config.extractorStyle:   
        if 'extractorStyle' in self.config:
            
            for _model in self.styleExtractors:
                _realStyleLogits, _realStyleFeatures, _realFidStyleFeature = \
                    _model.NetworkImplementation(inputImg=thisIO.inputs.imgReal, 
                                                 name_prefix=self.namePrefix+'/Style-'+_model.networkInfo.name,
                                                 is_training=False, reuse=reuse)
                thisIO.outputs.realStyleLogits.append(_realStyleLogits[1])
                thisIO.outputs.realStyleFeatures.append(_realStyleFeatures)
                thisIO.outputs.realStyleFidFeature.append(_realFidStyleFeature)
                if not reuse: 
                    styleExtractorSavers.append(_model.saver)
                    styleExtractorPaths.append(_model.networkInfo.path)
            
                _fakeStyleLogits, _fakeStyleFeatures, _fakeFidStyleFeature = \
                    _model.NetworkImplementation(inputImg=thisIO.inputs.imgFake, 
                                                 name_prefix=self.namePrefix+'/Style-'+_model.networkInfo.name,
                                                 is_training=False, reuse=True)   
                thisIO.outputs.fakeStyleLogits.append(_fakeStyleLogits[1]) 
                thisIO.outputs.fakeStyleFeatures.append(_fakeStyleFeatures)
                thisIO.outputs.fakeStyleFidFeature.append(_fakeFidStyleFeature)
        
        
        
        return thisIO, [contentExtractorSavers, styleExtractorSavers], [contentExtractorPaths, styleExtractorPaths]
        
