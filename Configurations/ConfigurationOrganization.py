# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.client import device_lib
import re, math, os, importlib
import numpy as np
from Utilities.utils import SplitName


from Utilities.file_operations import read_from_file


class NetworkConfigObject(object):
    def __init__(self, name, path, device):
        self.name=name
        self.path=path
        self.device=device
        self.network=self.name.split('-')[0]
        specifyDecoder=False
        if 'Decoder' in self.name:
            specifyDecoder=True
        
        if 'Encoder' in self.name or 'Mixer' in self.name or 'Decoder' in self.name:
            splits = self.name.split('-')
            for ii in splits:
                if 'Encoder' in ii:
                    self.encoder= ii
                    if not specifyDecoder:
                        self.decoder=ii
                if 'Mixer' in ii:
                    self.mixer=ii
                if 'Decoder' in ii:
                    self.decoder=ii
                    
        
        if hasattr(self, 'decoder'):
            self.decoder=self.decoder.replace('Encoder','Decoder')
            if not specifyDecoder:
                splits=SplitName(self.decoder)
                decoderNew = splits[0]
                for ii in range(len(splits)-1):
                    decoderNew = decoderNew+splits[-ii-1]
                self.decoder=decoderNew
            
        # if encoder is not None:
        #     self.encoder = encoder
        # # if encoderArchitecture is not None:
        #     self.encoder = encoderArchitecture
        
        


class DatasetConfigObject(object):
    class _dataPath(object):
        def __init__(self, txtPath, dataPath):
            self.txtPath=txtPath
            self.dataPath=dataPath
            # self.loadedLabel1Vec=-1
            # self.loadedLabel0Vec=-1
    class _trainAndTest(object):
        def __init__(self, train, test):
            self.train=train
            self.test=test
            
    class _ContentAndStyle(object):
        def __init__(self, content, style, different):
            self.content=content
            self.style=style
            self.different=different
            
    class _augmentation(object):
        def __init__(self, translation, rotation, flip):
            self.translation=translation
            self.rotation=rotation
            self.flip=flip
            self.randThick=False
                    
    def __init__(self, trainAugmentContentTranslation, trainAugmentContentRotation, trainAugmentContentFlip, 
                 trainAugmentStyleTranslation, trainAugmentStyleRotation, trainAugmentStyleFlip, 
                 testAugmentContentTranslation, testAugmentContentRotation, testAugmentContentFlip, 
                 testAugmentStyleTranslation, testAugmentStyleRotation, testAugmentStyleFlip, 
                 trainSplitContentStyleAugmentation,testSplitContentStyleAugmentation,
                 inputStyleNum, imgWidth, channels, label0VecTxt, label1VecTxt, dataRoot):
        
        trainContentAugmentation = self._augmentation(translation=trainAugmentContentTranslation,
                                                      rotation=trainAugmentContentRotation,
                                                      flip=trainAugmentContentFlip)
        testContentAugmentation = self._augmentation(translation=testAugmentContentTranslation,
                                                     rotation=testAugmentContentRotation,
                                                     flip=testAugmentContentFlip)
        trainStyleAugmentation = self._augmentation(translation=trainAugmentStyleTranslation,
                                                      rotation=trainAugmentStyleRotation,
                                                      flip=trainAugmentStyleFlip)
        testStyleAugmentation = self._augmentation(translation=testAugmentStyleTranslation,
                                                      rotation=testAugmentStyleRotation,
                                                      flip=testAugmentStyleFlip)
        trainPack = self._ContentAndStyle(content=trainContentAugmentation,
                                          style=trainStyleAugmentation,
                                          different=trainSplitContentStyleAugmentation)
        testPack = self._ContentAndStyle(content=testContentAugmentation,
                                          style=testStyleAugmentation,
                                          different=testSplitContentStyleAugmentation)
        self.augmentation = self._trainAndTest(train=trainPack, test=testPack)
        self.inputStyleNum=inputStyleNum
        self.imgWidth=imgWidth
        self.channels=channels
        self.loadedLabel0Vec=read_from_file(os.path.join(dataRoot, label0VecTxt))
        self.loadedLabel1Vec=read_from_file(os.path.join(dataRoot, label1VecTxt))
        
        
    

class UserInterfaceObj(object):
    def __init__(self, expID, expDir, logDir,  resumeTrain, imgDir, skipTest):
        self.expID=expID
        self.expDir=expDir
        self.logDir=logDir
        self.resumeTrain=resumeTrain
        self.skipTest=skipTest
        self.trainImageDir=imgDir

class TrainParamObject(object):
    def __init__(self, args, seed,optimizer, initTrainEpochs, finalLrPctg, debugMode):
        self.optimizer=optimizer
        self.initTrainEpochs=initTrainEpochs
        self.finalLrPctg=finalLrPctg
        self.epochs=args.epochs
        self.batchSize=args.batchSize
        self.initLr=args.initLr
        self.debugMode=debugMode
        self.seed=seed
        
        

class ParameterSetting(object):
    def __init__(self, config, args):
        
        self.config=config
        selectedDevices = self.CheckGPUs()
        avalialbe_cpu, \
        available_gpu = self.FindAvailableDevices()
        
        if set(avalialbe_cpu).difference(set(selectedDevices)) \
        and set(available_gpu).difference(set(selectedDevices)):
            print('ERROR in Devices')
            return None
        
        self.config.expID = self.GenerateExpID(args.encoder, args.mixer, args.decoder)
        self.config.trainModelDir = os.path.join(os.path.join(self.config.expDir, 'Ckpts'), self.config.expID)
        self.config.trainLogDir = os.path.join(os.path.join(self.config.expDir, 'Logs'), self.config.expID)
        self.config.trainImageDir = os.path.join(os.path.join(self.config.expDir, 'Images'), self.config.expID)
        
        
        #setting the user interface
        self.config.userInterface = UserInterfaceObj(expID=self.config.expID, 
                                                     expDir=self.config.trainModelDir, 
                                                     logDir=self.config.trainLogDir, 
                                                     imgDir=self.config.trainImageDir,
                                                     resumeTrain=args.resumeTrain,
                                                     skipTest=args.skipTest)
        # self.config.pop('expID', None)
        self.config.pop('expDir', None)
        self.config.pop('trainModelDir', None)
        self.config.pop('trainLogDir', None)
        self.config.pop('skipTest', None)
        
        
        
        # setting generator and discriminator
        self.config.generator = NetworkConfigObject(name=self.config.expID,
                                                    path=os.path.join(self.config.userInterface.expDir,'Generator'), 
                                                    device=self.config.generatorDevice)
        # self.config.generator = NetworkConfigObject(name=self.config.generator, 
        #                                             path=os.path.join(self.config.userInterface.expDir,'Generator'), 
        #                                             device=self.config.generatorDevice)
        self.config.pop('generatorDevice', None)
        
        
        self.config.discriminator = NetworkConfigObject(name=self.config.discriminator, 
                                                        path=os.path.join(self.config.userInterface.expDir,'Discriminator'), 
                                                        device=self.config.discrminatorDevice)
        self.config.pop('discrminatorDevice', None)
        
        # setting feature extractors
        if self.config.true_fake_target_extractor_dir:
            self.config.extractorTrueFake = NetworkConfigObject(name='TrueFakeFeatureExtractor', 
                                                          path=self.config.true_fake_target_extractor_dir, 
                                                          device=self.config.featureExtractorDevice)
        
        if self.config.content_prototype_extractor_dir:
            self.config.extractorContent = self.ProcessNetworks(self.config.content_prototype_extractor_dir)
            
        if self.config.style_reference_extractor_dir:
            self.config.extractorStyle = self.ProcessNetworks(self.config.style_reference_extractor_dir)
        self.config.pop('featureExtractorDevice', None)
        self.config.pop('true_fake_target_extractor_dir', None)
        self.config.pop('content_prototype_extractor_dir', None)
        self.config.pop('style_reference_extractor_dir', None)
        
        
        
        # # setting the discriminator as the last validator
        # self.config.ValidationContentModels.append(self.config.discriminator)
        # self.config.ValidationStyleModels.append(self.config.discriminator)
        
        
        
        # setting dataset object
        
        dataRootPath = importlib.import_module('.'+args.config, package='Configurations').dataPathRoot
        self.config.datasetConfig=DatasetConfigObject(trainAugmentContentTranslation=self.config.trainAugmentContentTranslation, 
                                                      trainAugmentContentRotation=self.config.trainAugmentContentRotation,
                                                      trainAugmentContentFlip=self.config.trainAugmentContentFlip,
                                                      trainAugmentStyleTranslation=self.config.trainAugmentStyleTranslation, 
                                                      trainAugmentStyleRotation=self.config.trainAugmentStyleRotation,
                                                      trainAugmentStyleFlip=self.config.trainAugmentStyleFlip,
                                                      testAugmentContentTranslation=self.config.testAugmentContentTranslation, 
                                                      testAugmentContentRotation=self.config.testAugmentContentRotation,
                                                      testAugmentContentFlip=self.config.testAugmentContentFlip,
                                                      testAugmentStyleTranslation=self.config.testAugmentStyleTranslation, 
                                                      testAugmentStyleRotation=self.config.testAugmentStyleRotation,
                                                      testAugmentStyleFlip=self.config.testAugmentStyleFlip,
                                                      trainSplitContentStyleAugmentation=self.config.trainSplitContentStyleAugmentation,
                                                      testSplitContentStyleAugmentation=self.config.testSplitContentStyleAugmentation,
                                                      inputStyleNum=self.config.inputStyleNum, 
                                                      imgWidth=self.config.imgWidth,
                                                      channels=self.config.channels,
                                                      dataRoot=dataRootPath,
                                                      label1VecTxt=self.config.FullLabel1Vec,
                                                      label0VecTxt=self.config.FullLabel0Vec)
        self.config.datasetConfig.contentData = \
            DatasetConfigObject._dataPath(txtPath=self.config.file_list_txt_content,
                                          dataPath=self.config.content_data_dir)
        self.config.datasetConfig.styleDataTrain = \
            DatasetConfigObject._dataPath(txtPath=self.config.file_list_txt_style_train, 
                                          dataPath=self.config.style_train_data_dir)
        self.config.datasetConfig.styleDataValidate = \
            DatasetConfigObject._dataPath(txtPath=self.config.file_list_txt_style_validation, 
                                          dataPath=self.config.style_validation_data_dir)
        
        self.config.pop('content_data_dir', None)
        self.config.pop('style_train_data_dir', None)
        self.config.pop('style_validation_data_dir', None)
        self.config.pop('file_list_txt_content', None)
        self.config.pop('file_list_txt_style_train', None)
        self.config.pop('file_list_txt_style_validation', None)
        
        self.config.pop('trainAugmentContentTranslation', None)
        self.config.pop('trainAugmentContentRotation', None)
        self.config.pop('trainAugmentContentFlip', None)
        self.config.pop('trainAugmentStyleTranslation', None)
        self.config.pop('trainAugmentStyleRotation', None)
        self.config.pop('trainAugmentStyleFlip', None)
        self.config.pop('testAugmentContentTranslation', None)
        self.config.pop('testAugmentContentRotation', None)
        self.config.pop('testAugmentContentFlip', None)
        self.config.pop('testAugmentStyleTranslation', None)
        self.config.pop('testAugmentStyleRotation', None)
        self.config.pop('testAugmentStyleFlip', None)
        self.config.pop('trainSplitContentStyleAugmentation', None)
        self.config.pop('testSplitContentStyleAugmentation', None)
        
        self.config.pop('inputStyleNum', None)
        self.config.pop('imgWidth', None)
        self.config.pop('channels', None)
        self.config.pop('label1VecTxt', None)
        self.config.pop('label0VecTxt', None)
        
        
        
        # setting the training parameters
        self.config.trainParams = TrainParamObject(args=args, 
                                                   seed=self.config.seed,
                                                   optimizer=self.config.optimization_method, 
                                                   initTrainEpochs=self.config.initTrainEpochs, 
                                                   finalLrPctg=self.config.final_learning_rate_pctg,
                                                   debugMode=self.config.debugMode)
        self.config.pop('optimization_method', None)
        self.config.pop('initTrainEpochs', None)
        self.config.pop('final_learning_rate_pctg', None)
        self.config.pop('debugMode', None)
        self.config.pop('seed', None)
        
        
        
        
        
    def ProcessNetworks(self, models):
        output=list()
        for ii in models:
            this_validation = ii[[ii.start() for ii in re.finditer("_",ii)][-1]+1:]
            this_name = this_validation[:this_validation.find('/')]
            this_device=ii[ii.find('@')+1:]
            this_path=ii[:ii.find('@')]
            this_network_obj = NetworkConfigObject(name=this_name,
                                                   device=this_device,
                                                   path=this_path)
            output.append(this_network_obj)
        return output

    
    
    def CheckGPUs(self):
        foundItems = list()
        for key, value in self.config.items():
            if isinstance(value, list):
                for ii in (range(len(value))):
                    _value = value[ii]
                    if not isinstance(_value, str):
                        break
                    _occuranceGPU = [m.start() for m in re.finditer('GPU', _value)]
                    _occuranceCPU = [m.start() for m in re.finditer('CPU', _value)]
                    
                    for jj in _occuranceGPU:
                        _item = _value[jj-8:jj+5]
                        if _item not in foundItems:
                            foundItems.append(_item)
                    for jj in _occuranceCPU:
                        _item = _value[jj-8:jj+5]
                        if _item not in foundItems:
                            foundItems.append(_item) 
            elif isinstance(value, str):
                _occuranceGPU = [m.start() for m in re.finditer('GPU', value)]
                _occuranceCPU = [m.start() for m in re.finditer('CPU', value)]
                for jj in _occuranceGPU:
                        _item = value[jj-8:jj+5]
                        if _item not in foundItems:
                            foundItems.append(_item)
                for jj in _occuranceCPU:
                    _item = value[jj-8:jj+5]
                    if _item not in foundItems:
                        foundItems.append(_item)     
                
                           
                
        return foundItems
    
    def FindAvailableDevices(self):
        local_device_protos = device_lib.list_local_devices()
        cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
        gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']

        for ii in range(len(gpu_device)):
            gpu_device[ii]=str(gpu_device[ii])
        for ii in range(len(cpu_device)):
            cpu_device[ii]=str(cpu_device[ii])

        print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
        print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
        gpu_device.sort()
        cpu_device.sort()

        return cpu_device, gpu_device

    def GenerateExpID(self, encoder, mixer, decoder):
        id = "Exp%s-%s-%s" % (self.config.expID, encoder, mixer)
        if decoder is not None:
            id = id+"-%s" % (decoder)
        # id = id + '-'+encoder+'_Encoder'
        if not self.config.discriminator=='NA':
            id=id+"-%s" % self.config.discriminator
        return id