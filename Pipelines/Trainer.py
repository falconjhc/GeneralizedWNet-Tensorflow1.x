# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function


GRAYSCALE_AVG = 127.5
TINIEST_LR = 0.00005
current_output_high_level_features=[1,2,3,4,5]
high_level_feature_penality_pctg=[0.1,0.15,0.2,0.25,0.3]
model_save_epochs=5
display_style_reference_num=4
import cv2

import sys
sys.path.append('..')
from random import choice

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import random as rnd
import os
import shutil
import time
from Pipelines.Dataset import DataProvider
import re
from Utilities.ops import fc
from Utilities.utils import scale_back_for_img, scale_back_for_dif, merge, correct_ckpt_path

from Utilities.utils import image_show


from Networks.Generators.WNetGeneratorBase import WNetGeneratorBase as Generator
# from Networks.Generators.VanillaWNetBase import VanillaWNetGenerator as VanillaWNet
# from Networks.Generators.TransWNetBase import TransWNetGenerator as TransWNet

from Networks.FeatureExtractor.FeatureExtractorBase import FeatureExtractor
from LossAccuracyEntropy.Loss import Loss
from LossAccuracyEntropy.AccuracyEntropy import AccuracyAndEntropy
from Pipelines.ModuleBase import ModuleBlock, EvaluationIO



# backBones = {
#     "PlainWNet": PlainWNet,
#     "VanillaWNet": VanillaWNet,
#     "TransWNet": TransWNet
# }

eps = 1e-9
RECORD_TIME=60
NUM_SAMPLE_PER_EPOCH=1000
RECORD_PCTG=NUM_SAMPLE_PER_EPOCH/100
DISP_VALIDATE_IMGS=10

class Trainer(object):

    # constructor
    def __init__(self, hyperParams=-1, penalties=-1):

        self.print_separater = "#################################################################"

        
        self.config=hyperParams
        self.penalties=penalties
        for key, value in self.penalties.items():
            if isinstance(value, list):
                (np.array(self.penalties[key])+eps).tolist()
            else:
                self.penalties[key] = value + eps
        
        self.config.initializer = 'XavierInit'
        self.accuracy_k=[1,3,5,10,20,50]
        

        # self.discriminator = discriminator_dict[self.config.discriminator.name]
    
        
        if self.config.userInterface.resumeTrain==0 and os.path.exists(self.config.userInterface.logDir):
            shutil.rmtree(self.config.userInterface.logDir)
        if self.config.userInterface.resumeTrain==0 and os.path.exists(self.config.userInterface.trainImageDir):
            shutil.rmtree(self.config.userInterface.trainImageDir)
        if self.config.userInterface.resumeTrain==0 and os.path.exists(self.config.userInterface.expDir):
            shutil.rmtree(self.config.userInterface.expDir)
        if not os.path.exists(self.config.userInterface.logDir):
            os.makedirs(self.config.userInterface.logDir)
        if not os.path.exists(self.config.userInterface.trainImageDir):
            os.makedirs(self.config.userInterface.trainImageDir)
        if not os.path.exists(self.config.userInterface.expDir):
            os.makedirs(os.path.join(self.config.userInterface.expDir,'Generator/ContentEncoder'))
            os.makedirs(os.path.join(self.config.userInterface.expDir,'Generator/StyleEncoder'))
            os.makedirs(os.path.join(self.config.userInterface.expDir,'Generator/Mixer'))
            os.makedirs(os.path.join(self.config.userInterface.expDir,'Generator/Decoder'))
            os.makedirs(os.path.join(self.config.userInterface.expDir,'Discriminator'))
            os.makedirs(os.path.join(self.config.userInterface.expDir,'Framework'))
            
        
        # init all the directories
        self.sess = None

            



    def SaveModel(self, saver, model_dir, global_step, model_name):
        step = global_step.eval(session=self.sess)
        if step==0:
            step=1
        print("Model saved @%s" %model_dir)
        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=int(step))

    
    def ModelRestore(self, saver, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        corrected_ckpt_path = correct_ckpt_path(real_dir=model_dir,

                                                maybe_path=ckpt.model_checkpoint_path)
        if ckpt:
            print(corrected_ckpt_path)
            saver.restore(self.sess, corrected_ckpt_path)
            # print("ModelRestored: ",end='')
            # print("@%s" % model_dir)
            print(self.print_separater)
            return True
        else:
            print("fail to restore model %s" % model_dir)
            print(self.print_separater)
            return False

    

    def CreateSummary(self, lr, ioLoss, ioAcryEtpy):
        
        def _buildSummary(_name, _value, summary):
            # _name = displayLossName[ii]%'Train'
            _thisSummary=tf.summary.scalar(_name, _value)
            summary=tf.summary.merge([summary, _thisSummary])
            return summary
        
        displayLossName={'L1': '01-LossReconstruction/L1-%s',
                         'content': '01-LossReconstruction/DeepPerceptualContentSum-%s',
                         'style': '01-LossReconstruction/DeepPerceptualStyleSum-%s',
                         'mseContent': '011-LossDeepPerceptual-ContentMSE/',
                         'mseStyle': '013-LossDeepPerceptual-StyleMSE/',
                         'vnContent': '012-LossDeepPerceptual-ContentVN/',
                         'vnStyle': '014-LossDeepPerceptual-StyleVN/',
                         'ConstContent': '01-LossGenerator/ConstContent-%s',
                         'ConstStyle': '01-LossGenerator/ConstStyle-%s',
                         'CategoryContentOnOrg': '01-LossGenerator/CategoryRealContent-%s',
                         'CategoryContentOnGenerated': '01-LossGenerator/CategoryFakeContent-%s',
                         'CategoryStyleOnOrg': '01-LossGenerator/CategoryRealStyle-%s',
                         'CategoryStyleOnGenerated': '01-LossGenerator/CategoryFakeStyle-%s',
                         'fidContent': '051-ContentFID/',
                         'fidStyle': '052-StyleFID/',
                         'fidContentSum':"05-FIDSum/Content-%s",
                         'fidStyleSum': "05-FIDSum/Style-%s"
                         }
        
        trnImages = tf.placeholder(tf.float32, [1, self.config.trainParams.batchSize * self.config.datasetConfig.imgWidth,
                                                self.config.datasetConfig.imgWidth * (self.config.datasetConfig.displayStyleNum+3+1), 3])
        valImages = tf.placeholder(tf.float32, [1, self.config.trainParams.batchSize * self.config.datasetConfig.imgWidth,
                                               self.config.datasetConfig.imgWidth * (self.config.datasetConfig.displayStyleNum+3+1), 3])

        summaryTrnImgs = tf.summary.image('TrainImages', trnImages)
        summaryValImgs = tf.summary.image('ValidationImages', valImages)
        summaryLr = tf.summary.scalar('00-LearningRate', lr)
        
        # Losses
        gLossSummariesTrain=[]
        gLossSummariesValidate=[]
        for ii in ioLoss.train.lossG:  
            gLossSummariesTrain=_buildSummary(_name=displayLossName[ii]%'Train', _value=ioLoss.train.lossG[ii], summary=gLossSummariesTrain)
        for ii in ioLoss.train.lossFE: 
            
            if not isinstance(ioLoss.train.lossFE[ii],list):
            #if not 'mse' in ii and not 'vn' in ii and not 'fid' in ii:
                gLossSummariesTrain=_buildSummary(_name=displayLossName[ii]%'Train', _value=ioLoss.train.lossFE[ii], summary=gLossSummariesTrain)
            else:
                for jj in range(len(ioLoss.train.lossFE[ii])):
                    # print(ii+'_'+displayLossName[ii])
                    if 'Content' in ii:
                        _name=displayLossName[ii]+self.config.extractorContent[jj].name+'-%s'%'Train'
                    elif 'Style' in ii:
                        _name=displayLossName[ii]+self.config.extractorStyle[jj].name+'-%s'%'Train'
                    gLossSummariesTrain=_buildSummary(_name=_name, _value=ioLoss.train.lossFE[ii][jj], summary=gLossSummariesTrain)
        
        
        
        for ii in ioLoss.testOnValidate.lossG: 
            gLossSummariesValidate=_buildSummary(_name=displayLossName[ii]%'Validate', _value=ioLoss.testOnValidate.lossG[ii], summary=gLossSummariesValidate)
        for ii in ioLoss.testOnValidate.lossFE: 
            if not isinstance(ioLoss.testOnValidate.lossFE[ii],list):
            #if not 'mse' in ii and not 'vn' in ii and not 'fid' in ii:
                gLossSummariesValidate=_buildSummary(_name=displayLossName[ii]%'Validate', _value=ioLoss.testOnValidate.lossFE[ii], summary=gLossSummariesValidate)
            else:
                for jj in range(len(ioLoss.train.lossFE[ii])):
                    #print(ii+'_'+displayLossName[ii])
                    if 'Content' in ii:
                        _name=displayLossName[ii]+self.config.extractorContent[jj].name+'-%s'%'Validate'
                    elif 'Style' in ii:
                        _name=displayLossName[ii]+self.config.extractorStyle[jj].name+'-%s'%'Validate'
                    gLossSummariesValidate=_buildSummary(_name=_name, _value=ioLoss.testOnValidate.lossFE[ii][jj], summary=gLossSummariesValidate)
        
        
        # Content Feature Extractor Accuracy
        contentAccuracySummary=[]
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.realContent)):
            contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Real-'+self.config.extractorContent[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.realContent[ii], 
                                                 summary=contentAccuracySummary)
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.fakeContent)):
            contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Fake-'+self.config.extractorContent[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.fakeContent[ii], 
                                                 summary=contentAccuracySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.realContent)):
            contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Real-'+self.config.extractorContent[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.realContent[ii], 
                                                 summary=contentAccuracySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.fakeContent)):
            contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Fake-'+self.config.extractorContent[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.fakeContent[ii], 
                                                 summary=contentAccuracySummary)
        
        # Content Generator Encoder Accuracy
        contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Real-'+'ContentEncoder'+'-Train', 
                                             _value=ioAcryEtpy.testOnTrain.generatorCategory.accuracy.contentReal, 
                                             summary=contentAccuracySummary)
        contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Fake-'+'ContentEncoder'+'-Train', 
                                             _value=ioAcryEtpy.testOnTrain.generatorCategory.accuracy.contentFake, 
                                             summary=contentAccuracySummary)
        contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Real-'+'ContentEncoder'+'-Validate', 
                                             _value=ioAcryEtpy.testOnValidate.generatorCategory.accuracy.contentReal, 
                                             summary=contentAccuracySummary)
        contentAccuracySummary=_buildSummary(_name='03-ContentAccuracy/Fake-'+'ContentEncoder'+'-Validate', 
                                             _value=ioAcryEtpy.testOnValidate.generatorCategory.accuracy.contentFake, 
                                             summary=contentAccuracySummary)
        
        # Style Feature Extractor Accuracy
        styleAccuracySummary=[]
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.realStyle)):
            styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Real-'+self.config.extractorStyle[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.realStyle[ii], 
                                                 summary=styleAccuracySummary)
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.fakeStyle)):
            styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Fake-'+self.config.extractorStyle[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.accuracy.fakeStyle[ii], 
                                                 summary=styleAccuracySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.realStyle)):
            styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Real-'+self.config.extractorStyle[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.realStyle[ii], 
                                                 summary=styleAccuracySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.fakeStyle)):
            styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Fake-'+self.config.extractorStyle[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.accuracy.fakeStyle[ii], 
                                                 summary=styleAccuracySummary)
        
        # Style Generator Encoder Accuracy
        styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Real-'+'StyleEncoder'+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.generatorCategory.accuracy.styleReal, 
                                                 summary=styleAccuracySummary)
        styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Fake-'+'StyleEncoder'+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.generatorCategory.accuracy.styleFake, 
                                                 summary=styleAccuracySummary)
        styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Real-'+'StyleEncoder'+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.generatorCategory.accuracy.styleReal, 
                                                 summary=styleAccuracySummary)
        styleAccuracySummary=_buildSummary(_name='03-StyleAccuracy/Fake-'+'StyleEncoder'+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.generatorCategory.accuracy.styleFake, 
                                                 summary=styleAccuracySummary)
            
        
        
        
        # Content Feature Extractor Entropy
        contentEntropySummary=[]
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.realContent)):
            contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Real-'+self.config.extractorContent[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.realContent[ii], 
                                                 summary=contentEntropySummary)
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.fakeContent)):
            contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Fake-'+self.config.extractorContent[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.fakeContent[ii], 
                                                 summary=contentEntropySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.realContent)):
            contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Real-'+self.config.extractorContent[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.realContent[ii], 
                                                 summary=contentEntropySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.fakeContent)):
            contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Fake-'+self.config.extractorContent[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.fakeContent[ii], 
                                                 summary=contentEntropySummary)
        
        # Content Generator Encoder Entropy
        contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Real-'+'ContentEncoder'+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.generatorCategory.entropy.contentReal, 
                                                 summary=contentEntropySummary)
        contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Fake-'+'ContentEncoder'+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.generatorCategory.entropy.contentFake, 
                                                 summary=contentEntropySummary)
        contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Real-'+'ContentEncoder'+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.generatorCategory.entropy.contentReal, 
                                                 summary=contentEntropySummary)
        contentEntropySummary=_buildSummary(_name='04-ContentEntropy/Fake-'+'ContentEncoder'+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.generatorCategory.entropy.contentFake, 
                                                 summary=contentEntropySummary)
        
        # Style Feature Extractor Entropy
        styleEntropySummary=[]
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.realStyle)):
            styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Real-'+self.config.extractorStyle[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.realStyle[ii], 
                                                 summary=styleEntropySummary)
        for ii in range(len(ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.fakeStyle)):
            styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Fake-'+self.config.extractorStyle[ii].name+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.featureExtractorCategory.entropy.fakeStyle[ii], 
                                                 summary=styleEntropySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.realStyle)):
            styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Real-'+self.config.extractorStyle[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.realStyle[ii], 
                                                 summary=styleEntropySummary)
        for ii in range(len(ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.fakeStyle)):
            styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Fake-'+self.config.extractorStyle[ii].name+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.featureExtractorCategory.entropy.fakeStyle[ii], 
                                                 summary=styleEntropySummary)
        
        # Style Generator Encoder Entropy
        styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Real-'+'StyleEncoder'+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.generatorCategory.entropy.styleReal, 
                                                 summary=styleEntropySummary)
        styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Fake-'+'StyleEncoder'+'-Train', 
                                                 _value=ioAcryEtpy.testOnTrain.generatorCategory.entropy.styleFake, 
                                                 summary=styleEntropySummary)
        styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Real-'+'StyleEncoder'+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.generatorCategory.entropy.styleReal, 
                                                 summary=styleEntropySummary)
        styleEntropySummary=_buildSummary(_name='04-StyleEntropy/Fake-'+'StyleEncoder'+'-Validate', 
                                                 _value=ioAcryEtpy.testOnValidate.generatorCategory.entropy.styleFake, 
                                                 summary=styleEntropySummary)
        
        
        
        
        # Full Validation Summaries
        validationContentSummariesRealTrain = []
        validationContentValueRealTrain = []
        for ii in self.config.extractorContent:
            _name = '02-TestFullSetAcry-Content/'+ ii.name+'-Real-Train'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationContentSummariesRealTrain.append(_summary)
            validationContentValueRealTrain.append(_value)
        
        validationStyleSummariesRealTrain = []
        validationStyleValueRealTrain = []
        for ii in self.config.extractorStyle:
            _name = '02-TestFullSetAcry-Style/'+ ii.name+'-Real-Train'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationStyleSummariesRealTrain.append(_summary)
            validationStyleValueRealTrain.append(_value)
        
        
        validationContentSummariesRealValidation = []
        validationContentValueRealValidation = []
        for ii in self.config.extractorContent:
            _name = '02-TestFullSetAcry-Content/'+ ii.name+'-Real-Validation'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationContentSummariesRealValidation.append(_summary)
            validationContentValueRealValidation.append(_value)
        
        validationStyleSummariesRealValidation = []
        validationStyleValueRealValidation = []
        for ii in self.config.extractorStyle:
            _name = '02-TestFullSetAcry-Style/'+ ii.name+'-Real-Validation'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationStyleSummariesRealValidation.append(_summary)
            validationStyleValueRealValidation.append(_value)
            
        
        validationContentSummariesFakeTrain = []
        validationContentValueFakeTrain = []
        for ii in self.config.extractorContent:
            _name = '02-TestFullSetAcry-Content/'+ ii.name+'-Fake-Train'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationContentSummariesFakeTrain.append(_summary)
            validationContentValueFakeTrain.append(_value)
        
        validationStyleSummariesFakeTrain = []
        validationStyleValueFakeTrain = []
        for ii in self.config.extractorStyle:
            _name = '02-TestFullSetAcry-Style/'+ ii.name+'-Fake-Train'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationStyleSummariesFakeTrain.append(_summary)
            validationStyleValueFakeTrain.append(_value)
        
        
        validationContentSummariesFakeValidation = []
        validationContentValueFakeValidation = []
        for ii in self.config.extractorContent:
            _name = '02-TestFullSetAcry-Content/'+ ii.name+'-Fake-Validation'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationContentSummariesFakeValidation.append(_summary)
            validationContentValueFakeValidation.append(_value)
        
        validationStyleSummariesFakeValidation = []
        validationStyleValueFakeValidation = []
        for ii in self.config.extractorStyle:
            _name = '02-TestFullSetAcry-Style/'+ ii.name+'-Fake-Validation'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary=tf.summary.scalar(_name, _value)
            validationStyleSummariesFakeValidation.append(_summary)
            validationStyleValueFakeValidation.append(_value)
            
            
            
        # Full FID summaries
        fidContentTrainSummary=[]
        fidContentTestSummary=[]
        fidStyleTrainSummary=[]
        fidStyleTestSummary=[]
        fidContentTrainValue=[]
        fidContentTestValue=[]
        fidStyleTrainValue=[]
        fidStyleTestValue=[]
        for ii in self.config.extractorContent:
            _name = '02-TestFullSetFID-Content/'+ ii.name+'-Train'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary = tf.summary.scalar(_name, _value)
            fidContentTrainSummary.append(_summary)
            fidContentTrainValue.append(_value)
            
            _name = '02-TestFullSetFID-Content/'+ ii.name+'-Validation'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary = tf.summary.scalar(_name, _value)
            fidContentTestSummary.append(_summary)
            fidContentTestValue.append(_value)
            
        for ii in self.config.extractorStyle:
            _name = '02-TestFullSetFID-Style/'+ ii.name+'-Train'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary = tf.summary.scalar(_name, _value)
            fidStyleTrainSummary.append(_summary)
            fidStyleTrainValue.append(_value)
            
            _name = '02-TestFullSetFID-Style/'+ ii.name+'-Validation'
            _value = tf.placeholder(tf.float32, name=_name+'-ValuePH')
            _summary = tf.summary.scalar(_name, _value)
            fidStyleTestSummary.append(_summary)
            fidStyleTestValue.append(_value)
    
            
        # Full Validation on the Generator Encoders
        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Content/GeneratorEncoder-Real-Train'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Content/GeneratorEncoder-Real-Train', _value)
        validationContentSummariesRealTrain.append(_summary)
        validationContentValueRealTrain.append(_value)
        
        
        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Content/GeneratorEncoder-Fake-Train'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Content/GeneratorEncoder-Fake-Train', _value)
        validationContentSummariesFakeTrain.append(_summary)
        validationContentValueFakeTrain.append(_value)
        
        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Style/GeneratorEncoder-Real-Train'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Style/GeneratorEncoder-Real-Train', _value)
        validationStyleSummariesRealTrain.append(_summary)
        validationStyleValueRealTrain.append(_value)
        
        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Style/GeneratorEncoder-Fake-Train'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Style/GeneratorEncoder-Fake-Train', _value)
        validationStyleSummariesFakeTrain.append(_summary)
        validationStyleValueFakeTrain.append(_value)
        
        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Content/GeneratorEncoder-Real-Validation'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Content/GeneratorEncoder-Real-Validation', _value)
        validationContentSummariesRealValidation.append(_summary)
        validationContentValueRealValidation.append(_value)

        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Content/GeneratorEncoder-Fake-Validation'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Content/GeneratorEncoder-Fake-Validation', _value)
        validationContentSummariesFakeValidation.append(_summary)
        validationContentValueFakeValidation.append(_value)

        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Style/GeneratorEncoder-Real-Validation'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Style/GeneratorEncoder-Real-Validation', _value)
        validationStyleSummariesRealValidation.append(_summary)
        validationStyleValueRealValidation.append(_value)

        _value=tf.placeholder(tf.float32, name='02-TestFullSetAcry-Style/GeneratorEncoder-Fake-Validation'+'-ValuePH')
        _summary=tf.summary.scalar('02-TestFullSetAcry-Style/GeneratorEncoder-Fake-Validation',_value)
        validationStyleSummariesFakeValidation.append(_summary)
        validationStyleValueFakeValidation.append(_value)
        
        print("Tensorboard Summaries created")
        print(self.print_separater)
        
        
        
        return summaryLr, \
            tf.summary.merge([gLossSummariesTrain, gLossSummariesValidate]), \
            tf.summary.merge([contentAccuracySummary,styleAccuracySummary]), \
            tf.summary.merge([contentEntropySummary,styleEntropySummary]), \
            [trnImages, valImages, summaryTrnImgs, summaryValImgs], \
            [[validationContentSummariesRealTrain, validationContentSummariesFakeTrain, \
                validationStyleSummariesRealTrain, validationStyleSummariesFakeTrain, \
                    fidContentTrainSummary, fidStyleTrainSummary], 
            [validationContentValueRealTrain, validationContentValueFakeTrain, \
                validationStyleValueRealTrain, validationStyleValueFakeTrain, \
                    fidContentTrainValue, fidStyleTrainValue]],\
            [[validationContentSummariesRealValidation, validationContentSummariesFakeValidation, \
                validationStyleSummariesRealValidation, validationStyleSummariesFakeValidation, \
                    fidContentTestSummary, fidStyleTestSummary], 
            [validationContentValueRealValidation, validationContentValueFakeValidation, \
                validationStyleValueRealValidation, validationStyleValueFakeValidation, \
                    fidContentTestValue, fidStyleTestValue]]
    
    

    def BuildPipelineFramework(self):
        # for model base frameworks
        with tf.device('/device:CPU:0'):
            global_step = tf.get_variable('global_step',
                                          [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False,
                                          dtype=tf.int32)
            epoch_step = tf.get_variable('epoch_step',
                                         [],
                                         initializer=tf.constant_initializer(0),
                                         trainable=False,
                                         dtype=tf.int32)
            # epoch_step_increase_one_op = tf.assign(epoch_step, epoch_step + 1)
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")

            framework_var_list = list()
            framework_var_list.append(global_step)
            framework_var_list.append(epoch_step)
            
        epochIncrementOp = tf.assign(epoch_step, epoch_step + 1)

        saver_frameworks = tf.train.Saver(max_to_keep=model_save_epochs, var_list=framework_var_list)


        print("Framework built @%s." % '/device:CPU:0')
        return learning_rate, global_step, epoch_step, saver_frameworks, epochIncrementOp


    

    def CreateOptimizer(self, config, learning_rate,step, loss, gVars):
        gVars = gVars[0]+gVars[1]+gVars[2]+gVars[3]
        gOps= [ii for ii in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'Generator' in ii.name]
        with tf.control_dependencies([tf.group(*gOps)]):
            if config.trainParams.optimizer=='adam':
                optmG = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss.sumLossG+loss.sumLossFE,
                                                                                  var_list=gVars,
                                                                                  global_step=step)
            elif config.trainParams.optimizer=='SGD':
                optmG = tf.train.GradientDescentOptimizer(learning_rate, beta1=0.5).minimize(loss.sumLossG+loss.sumLossFE,
                                                                                             var_list=gVars,
                                                                                             global_step=step)
            print("Optimizer for the generator created: %s" % config.trainParams.optimizer)
            print(self.print_separater)
        
        return optmG


    def Initialization(self, generator, saver_frameworks, featureExtractors):
        # initialization of all the variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        print("Randomarization of all the weights completed")
        
        if self.config.userInterface.resumeTrain==1:
            # restore framework
            self.ModelRestore(saver=saver_frameworks,
                              model_dir=os.path.join(self.config.userInterface.expDir,'Framework'))
            
            # restore trained generator
            self.ModelRestore(saver=generator.savers[0],
                              model_dir=generator.path[0])
            self.ModelRestore(saver=generator.savers[1],
                              model_dir=generator.path[1])
            if not generator.savers[2]: # For mixer
                self.ModelRestore(saver=generator.savers[2],
                                model_dir=generator.path[2])
            self.ModelRestore(saver=generator.savers[3],
                              model_dir=generator.path[3])
            
            
        
        # restore feature extractors
        [contentExtractorSavers, styleExtractorSavers] = featureExtractors.savers
        [contentExtractorPath, styleExtractorPath] = featureExtractors.path
        
        if not len(styleExtractorSavers)==len(styleExtractorPath):
            print("ERROR")
        #namePrefix='FeatureExtractor/Style-'
        for ii in range(len(styleExtractorSavers)):
            _saver=styleExtractorSavers[ii]
            _path=styleExtractorPath[ii]
            self.ModelRestore(saver=_saver,
                              model_dir=_path)
        
        
        
        if not len(contentExtractorSavers)==len(contentExtractorPath):
            print("ERROR")
        #namePrefix='FeatureExtractor/Content-'
        for ii in range(len(contentExtractorSavers)):
            _saver=contentExtractorSavers[ii]
            _path=contentExtractorPath[ii]
            self.ModelRestore(saver=_saver,
                               model_dir=_path)
        print(self.print_separater)
        
        

    def Pipelines(self):
        
        self.trainStartTime=time.time()

        # if self.config.trainParams.debugMode == 1:
        #     RECORD_TIME= 5
        

        # with tf.Graph().as_default():

        # tensorflow parameters
        # DO NOT MODIFY!!!
        
        tf.random.set_random_seed(self.config.trainParams.seed)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        gpu_options.allocator_type = 'BFC'
        gpu_options.allow_growth = True
        # gpu_options.allow_software_placement = True
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)

        # config = tf.ConfigProto(per_process_gpu_memory_fraction=0.9)
        # self.sess = tf.Session(config=config)


        # define the data set 准备数据
        data_provider = DataProvider(config=self.config)


        if self.config.datasetConfig.loadedLabel1Vec == -1:
            self.config.datasetConfig.loadedLabel1Vec = data_provider.full_label1_vec
            

        self.involved_label0_list, self.involved_label1_list = data_provider.get_involved_label_list()
        self.content_input_num = data_provider.content_input_num
        # self.display_style_reference_num = np.min([display_style_reference_num, self.config.datasetConfig.inputStyleNum])
        
        self.content_input_number_actual = self.content_input_num
        self.display_content_reference_num = np.min([4, self.content_input_number_actual])

        # ignore
        delete_items=list()
        involved_label_list = self.involved_label1_list
        for ii in self.accuracy_k:
            if ii>len(involved_label_list):
                delete_items.append(ii)
        for ii in delete_items:
            self.accuracy_k.remove(ii)
        if delete_items and (not self.accuracy_k[len(self.accuracy_k)-1] == len(involved_label_list)):
            self.accuracy_k.append(len(involved_label_list))

        self.train_data_repeat_time = 1
        learning_rate_decay_rate = np.power(self.config.trainParams.finalLrPctg, 1.0 / (self.config.trainParams.epochs - 1))

        


        #######################################################################################
        #######################################################################################
        #                                model building
        #######################################################################################
        #######################################################################################

        # 构建迭代计数器+学习率
        learning_rate, \
        global_step, \
        epoch_step, \
        saver_frameworks, epochIncrementOp =\
            self.BuildPipelineFramework()


        # 构建模型
        # register the generator 构建生成器
        generatorBuider = Generator(dataLoader=data_provider,
                                             config=self.config,
                                             penalties=self.penalties)
        generator = ModuleBlock()
        generator.io.train, generator.savers, generator.trainVars, generator.path =\
            generatorBuider.BuildGenerator(isTraining=True, saveEpochs=model_save_epochs) # Build the training graph
        generator.io.testOnTrain, _,_,_=\
            generatorBuider.BuildGenerator(isTraining=False, validateOn='Trainset') # build the generator graph to validate the model on the train set
        generator.io.testOnValidate, _, _,_=\
            generatorBuider.BuildGenerator(isTraining=False, validateOn='Validateset') # build the generator graph to validate the model on the validate set
        print(self.print_separater)
        
        # register the feature extractor 构建特征提取
        featureExtractorBuider=FeatureExtractor(config=self.config,
                                                penalties=self.penalties,
                                                dataLoader=data_provider)
        featureExtractor = ModuleBlock()
        featureExtractor.io.train, featureExtractor.savers, featureExtractor.path = \
            featureExtractorBuider.BuildFeatureExtractor(generatorIO=generator.io.train, 
                                                    validateOn='Trainset', reuse=False, 
                                                    isTrain=True) # build the feature extractor graph to train the model
        featureExtractor.io.testOnTrain, _, _ = \
            featureExtractorBuider.BuildFeatureExtractor(generatorIO=generator.io.testOnTrain, 
                                                    validateOn='Trainset', reuse=True) # build the feature extractor graph to validate the train set
        featureExtractor.io.testOnValidate, _, _ = \
            featureExtractorBuider.BuildFeatureExtractor(generatorIO=generator.io.testOnValidate, 
                                                    validateOn='Validateset', reuse=True) # build the feature extractor graph to validate the validate set
        print(self.print_separater)
        
        
        
        
        #register the loss 构建loss损失函数
        lossBuilder = Loss(penalties=self.penalties)
        loss = EvaluationIO()
        loss.train=\
            lossBuilder.BuildLosses(generatorIO=generator.io.train,
                              featureExtractorIO=featureExtractor.io.train,
                              validateOn='Trainset', isTrain=True)
        loss.testOnTrain=\
            lossBuilder.BuildLosses(generatorIO=generator.io.testOnTrain,
                              featureExtractorIO=featureExtractor.io.testOnTrain,
                              validateOn='Trainset')
        loss.testOnValidate=\
            lossBuilder.BuildLosses(generatorIO=generator.io.testOnValidate,
                              featureExtractorIO=featureExtractor.io.testOnValidate,
                              validateOn='Validateset')
        print(self.print_separater)
        
        
        
        # register the accuracy calculation 构建计算Accuracy and Entropy
        accuracyEntropyBuilder = AccuracyAndEntropy()
        accuracyEntropy = EvaluationIO()
        accuracyEntropy.testOnTrain=\
            accuracyEntropyBuilder.BuildAccuracy(generatorIO=generator.io.testOnTrain,
                               featureExtractorIO=featureExtractor.io.testOnTrain,
                               validateOn='Trainset')
        accuracyEntropy.testOnValidate=\
            accuracyEntropyBuilder.BuildAccuracy(generatorIO=generator.io.testOnValidate,
                               featureExtractorIO=featureExtractor.io.testOnValidate,
                               validateOn='Validateset')
        print(self.print_separater)
        
        
        # create the optimizer 构建优化器
        optmG = self.CreateOptimizer(config=self.config, 
                                     learning_rate=learning_rate,
                                     step=global_step, 
                                     loss=loss.train,
                                     gVars=generator.trainVars)
        print(self.print_separater)
        
        
        # create tensorboard summaries 构建tensorboard summary
        summaryLR, summaryG, summaryAccuracy, summaryEntropy, summaryImages, \
            fullValidationTrainSummaries,  fullValidationValidateSummaries = \
                self.CreateSummary(lr=learning_rate,  ioLoss=loss,  ioAcryEtpy=accuracyEntropy)
        print(self.print_separater)
        summary_writer = tf.summary.FileWriter(self.config.userInterface.logDir, self.sess.graph)
        
        # model initialization
        self.Initialization(generator=generator, 
                            saver_frameworks=saver_frameworks,
                            featureExtractors=featureExtractor)
        
    
        # start training preparations
        print("%d Threads to read the data" % (data_provider.thread_num))
        print("BatchSize:%d, EpochNum:%d, LearningRateDecay:%.10f Per Epoch"
                % (self.config.trainParams.batchSize, self.config.trainParams.epochs, learning_rate_decay_rate))
        print("TrainingSize:%d, ValidateSize:%d, StyleLabel0_Vec:%d, StyleLabel1_Vec:%d" %
                (len(data_provider.train_iterator.true_style.data_list),
                len(data_provider.validate_iterator.true_style.data_list),
                len(self.involved_label0_list),
                len(self.involved_label1_list)))
        print("ContentLabel0_Vec:%d, ContentLabel1_Vec:%d" % (len(data_provider.content_label0_vec),len(data_provider.content_label1_vec)))
        # print("PrintInfo:%d secs"%(RECORD_TIME))
        print("InvolvedLabel0:%d, InvolvedLabel1:%d" % (len(self.involved_label0_list),
                                                        len(self.involved_label1_list)))
        # print("DataAugment/Flip:%d/%d, InputStyleNum:%d" % (self.config.datasetConfig.translation, self.config.datasetConfig.flip, self.config.datasetConfig.inputStyleNum))
        print(self.print_separater)
        print("Penalties:")
        print("Generator: PixelL1:%.3f,ConstCP/SR:%.3f/%.3f,Wgt:%.6f, BatchDist:%.5f;"
                % (self.penalties.Pixel_Reconstruction_Penalty,
                    self.penalties.Lconst_content_Penalty,
                    self.penalties.Lconst_style_Penalty,
                    self.penalties.generator_weight_decay_penalty,
                    self.penalties.Batch_StyleFeature_Discrimination_Penalty))
        print("Discriminator: Cat:%.3f,Dis:%.3f,WST-Grdt:%.3f,Wgt:%.6f;" % (self.penalties.Discriminator_Categorical_Penalty,
                                                                            self.penalties.Discriminative_Penalty,
                                                                            self.penalties.Discriminator_Gradient_Penalty,
                                                                            self.penalties.discriminator_weight_decay_penalty))
        print("Penalties of ContentFeatureExtractors: ", end='')
        for ii in self.config.extractorContent:
            print(ii.name+', ', end='')
        print(self.penalties.FeatureExtractorPenalty_ContentPrototype)
        # print(self.print_separater)
        print("Penalties of StyleFeatureExtractor: ", end='')
        for ii in self.config.extractorStyle:
            print(ii.name+', ', end='')
        print(self.penalties.FeatureExtractorPenalty_StyleReference)
        

        print("InitLearningRate:%.10f" % self.config.trainParams.initLr)
        #print("AdaIN_Mode:%s" % self.config.generator.adain)
        print(self.print_separater)
        print("Initialization completed, and training started right now.")


        print(self.print_separater)
        print(self.print_separater)
        print(self.print_separater)
        
        
        
        if self.config.userInterface.resumeTrain==1:
            eiStart = epoch_step.eval(self.sess)
            currentLr = self.config.trainParams.initLr * np.power(learning_rate_decay_rate, eiStart)

        else:
            eiStart = 0
            currentLr = self.config.trainParams.initLr
        training_epoch_list = range(eiStart,self.config.trainParams.epochs,1)      
            
        global_step_start = global_step.eval(session=self.sess)
        print("InitTrainingEpochs:%d" % (self.config.trainParams.initTrainEpochs))
        print("TrainingStart:Epoch:%d, GlobalStep:%d, LearnRate:%.5f" % (eiStart+1,global_step_start+1, currentLr))
        print("ContentLabel1Vec:")
        print(data_provider.content_label1_vec)   
        
        
        
        # Training Epochs
        for ei in training_epoch_list:
            init_val=False
            if ei == eiStart: init_val=True
            data_provider.dataset_reinitialization(sess=self.sess, init_for_val=init_val,
                                                   info_interval=RECORD_TIME/10)
            
            
        
            trainItrs = data_provider.compute_total_train_batch_num()
            valItrs = data_provider.compute_total_validate_batch_num()
            
            # Validate at the start
            if ei == eiStart:
                checkTrainImg=self.GenerateTensorboardImage(generatorIO=generator.io.testOnTrain)
                checkValidateImg=self.GenerateTensorboardImage(generatorIO=generator.io.testOnValidate)
                saveImage = np.squeeze(np.concatenate([checkTrainImg, checkValidateImg], axis=2))
                saveNamePath=os.path.join(self.config.userInterface.trainImageDir,'ImageAtEpoch%d-Iter%d.png' % (ei, global_step.eval(session=self.sess)))
                cv2.imwrite(saveNamePath, saveImage*255)
                print("Image Saved @ %s" % saveNamePath)    
                print(self.print_separater)
                
                if not self.config.userInterface.skipTest:
                    self.ValidateOneEpoch(inputIO=accuracyEntropy.testOnValidate,  iter_num=valItrs, print_info="Test@Val-Ep:%d" % (ei), 
                                          progress_info='NA', summaryOp=fullValidationValidateSummaries, 
                                          summaryWriter=summary_writer, ei=epoch_step, evalFID=True)
                    self.ValidateOneEpoch(inputIO=accuracyEntropy.testOnTrain,  iter_num=trainItrs, print_info="Test@Trns-Ep:%d" % (ei), 
                                          progress_info='NA', summaryOp=fullValidationTrainSummaries, 
                                          summaryWriter=summary_writer, ei=epoch_step, evalFID=True)
                    
                
            
            if not ei == eiStart:
                updateLr = currentLr * learning_rate_decay_rate
                print("decay learning rate from %.7f to %.7f" % (updateLr, updateLr))
                print(self.print_separater)
                currentLr = updateLr

            print(self.print_separater)
            print("Training @ Epoch:%d/%d with %d Iters is now commencing" % (ei + 1, self.config.trainParams.epochs, trainItrs))
            print(self.print_separater)
            
            progress_info = \
                self.TrainOneEpoch(ei=ei+1, eiStart=eiStart, 
                                   generator=generator,
                                   optmzG=optmG, 
                                   trainItrs=trainItrs, thisLR=currentLr, 
                                   global_step=global_step,
                                   lrPH=learning_rate, globalStepPH=global_step,
                                   summaryScalars=tf.summary.merge([summaryLR, summaryG, summaryAccuracy, summaryEntropy]), 
                                   summaryImgs = summaryImages, 
                                   summaryWriter=summary_writer)
            
            # validate in the end of each epoch
            self.sess.run(epochIncrementOp)
            print(self.print_separater)
            print("Time: %s, Checkpoint: SaveCheckpoint@step: %d" % (time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime()), 
                                                                     global_step.eval(session=self.sess)))
            self.SaveModel(saver=generator.savers[0], model_dir=generator.path[0],
                           global_step=global_step, model_name='ContentEncoder')
            self.SaveModel(saver=generator.savers[1], model_dir=generator.path[1],
                           global_step=global_step, model_name='StyleEncoder')
            if not generator.savers[2]:
                self.SaveModel(saver=generator.savers[2], model_dir=generator.path[2],
                            global_step=global_step, model_name='Mixer')
            self.SaveModel(saver=generator.savers[3], model_dir=generator.path[3],
                           global_step=global_step, model_name='Decoder')
            self.SaveModel(saver=saver_frameworks, model_dir=os.path.join(self.config.userInterface.expDir, 'Framework'),
                           global_step=global_step, model_name='Framework')
            
            if not self.config.userInterface.skipTest:
                if (ei+1)%10==0 or ei==training_epoch_list[-1]:
                    evalFid=True
                else:
                    evalFid=False
                # evalFid=False
                if ((ei+1) % 5 == 0 or (ei+1) <= 5):
                    self.ValidateOneEpoch(inputIO=accuracyEntropy.testOnTrain,  iter_num=trainItrs, print_info="Test@Trn-Ep:%d" % (ei+1), 
                                        progress_info=progress_info, summaryOp=fullValidationTrainSummaries, summaryWriter=summary_writer, ei=epoch_step, evalFID=evalFid)
                self.ValidateOneEpoch(inputIO=accuracyEntropy.testOnValidate,  iter_num=valItrs, print_info="Test@Val-Ep:%d" % (ei+1), 
                                    progress_info=progress_info, summaryOp=fullValidationValidateSummaries, summaryWriter=summary_writer, ei=epoch_step, evalFID=evalFid)
            print(self.print_separater)
        
        print("Training Completed. Good Luck. ")

        
        
    def TrainOneEpoch(self, ei, eiStart, optmzG, trainItrs, thisLR, lrPH, globalStepPH, generator,global_step,
                      discriminator=None, 
                      summaryScalars=None, summaryImgs=None, summaryWriter=None):
        
        # thisEpochStart = time.time()
        thisRoundStartTime = time.time()
        thisRoundStartItr = 0
        # trainItrs=50
        for bid in range(trainItrs):
            thisItrStart = time.time()
            info=""
            
            if generator.trainVars:
                self.sess.run(optmzG, feed_dict={lrPH:thisLR})
                info = info + "LR@G:%f" % thisLR
            # if discriminator.trainVars:
            #     d_lr_expansion=3.5
            #     a=1
                
            timeThisIter = time.time() - thisItrStart
            if self.config.userInterface.resumeTrain and ei==eiStart+1:
                timeFromStart = globalStepPH.eval(session=self.sess) * timeThisIter
            else:
                timeFromStart = time.time()- self.trainStartTime
            
            # Log Process
            # if bid%2==0:
            # if bid*100.0/(trainItrs*1.0)-thisRoundStartItr>RECORD_PCTG or bid == 0 or bid == trainItrs-1:
            if bid*float(NUM_SAMPLE_PER_EPOCH)/(trainItrs*1.0)-thisRoundStartItr>RECORD_PCTG or bid == 0 or bid == trainItrs-1:
                thisRoundStartTime=time.time()
                thisRoundStartItr =bid*float(NUM_SAMPLE_PER_EPOCH)/(trainItrs*1.0)
                current_time = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime())
                print(self.config.userInterface.expID)
                print("Training Time: %s, Epoch: %d/%d, Itr: %d/%d;" % (current_time, ei, 
                                                                        self.config.trainParams.epochs, bid + 1, trainItrs))
                
                print("ItrDuration: %.2fses, FullDuration: %.2fhrs (%.2fdays);" %
                          (timeThisIter, timeFromStart / 3600, timeFromStart / (3600 * 24)))
                
                pctgCompleted = float(globalStepPH.eval(session=self.sess)) / float((self.config.trainParams.epochs) * trainItrs) * 100
                pctgRemained = 100 - pctgCompleted
                hrs_estimated_remaining = (float(timeFromStart) / (pctgCompleted + eps)) * pctgRemained / 3600
                progress_info = "Pctg: %.2f%%, Estm: %.2fhrs (%.2fdays)" %\
                    (pctgCompleted, hrs_estimated_remaining, hrs_estimated_remaining / 24)
                print(progress_info)
                print("TrainingInfo: %s" % info)
                # print(self.print_separater)
                

                for ii in range(DISP_VALIDATE_IMGS):
                    checkValidateImg=self.GenerateTensorboardImage(generatorIO=generator.io.testOnValidate)
                    summaryWriter.add_summary(self.sess.run(summaryImgs[3], feed_dict={summaryImgs[1]: checkValidateImg}),   
                                              int(float(globalStepPH.eval(session=self.sess))/float(trainItrs)*NUM_SAMPLE_PER_EPOCH+ii))
                
                # image tensorboard log
                checkTrainImg=self.GenerateTensorboardImage(generatorIO=generator.io.testOnTrain)
                summaryWriter.add_summary(self.sess.run(summaryImgs[2], feed_dict={summaryImgs[0]: checkTrainImg}),  
                                          int(float(globalStepPH.eval(session=self.sess))/float(trainItrs)*NUM_SAMPLE_PER_EPOCH))
                
                if bid == trainItrs-1:
                    saveImage = np.squeeze(np.concatenate([checkTrainImg, checkValidateImg], axis=2))
                    saveNamePath=os.path.join(self.config.userInterface.trainImageDir,'ImageAtEpoch%d-Iter%d.png' % (ei, global_step.eval(session=self.sess)))
                    cv2.imwrite(saveNamePath, saveImage*255)
                    print("Image Saved @ %s" % saveNamePath)    
                
                # scalar tensorboard log
                summaryWriter.add_summary(self.sess.run(summaryScalars, feed_dict={lrPH: thisLR}), 
                                          int(float(globalStepPH.eval(session=self.sess))/float(trainItrs)*NUM_SAMPLE_PER_EPOCH))
                summaryWriter.flush()
                print("Tensorboard Logs have been updated, elapsed %.2f secs." % (time.time()-thisRoundStartTime))
                print(self.print_separater)
                
        return progress_info
                
                
            
    def GenerateTensorboardImage(self, generatorIO):
        
        # train mode
        styles, reals, fakes, content = \
            self.sess.run([generatorIO.inputs.displayStyles,
                           generatorIO.groundtruths.trueCharacter,
                           generatorIO.outputs.generated,
                           generatorIO.inputs.contents[0]])
            
        #contents = scale_back_for_img(images=contents[0])
        selectedContentIdx=choice(list(range(content.shape[-1])))
        content=np.expand_dims(content[:,:,:,selectedContentIdx], axis=-1)
        reals = scale_back_for_img(images=reals)
        fakes = scale_back_for_img(images=fakes)
        
        
        content = scale_back_for_img(images=content)
        
        
        
        for ii in range(len(styles)):
            _thisStyle=scale_back_for_img(images=styles[ii])
            if ii == 0:
                styleOut = _thisStyle
            else:
                styleOut=np.concatenate([styleOut, _thisStyle], axis=3)
        styles=styleOut
            
        reals=merge(reals, [self.config.trainParams.batchSize,1])
        fakes=merge(fakes, [self.config.trainParams.batchSize,1])
        content=merge(content, [self.config.trainParams.batchSize,1])
        difference = scale_back_for_dif(reals-fakes)
        
        
        for ii in range(styles.shape[-1]):
            thisStyle = merge(np.expand_dims(styles[:,:,:,ii], axis=3), [self.config.trainParams.batchSize,1])
            if ii ==0: 
                newStyle=thisStyle
            else: 
                newStyle=np.concatenate([newStyle, thisStyle], axis=1)
        styles=newStyle
        #dispImg = np.expand_dims(np.concatenate([contents, fakes, difference, reals, styles], axis=1), axis=0)
        dispImg = np.expand_dims(np.concatenate([styles,reals,fakes, difference, content], axis=1), axis=0)
        return dispImg
            
        
        
        
    def ValidateOneEpoch(self, inputIO,  iter_num, print_info, progress_info, summaryOp, summaryWriter, ei, evalFID=False):
        print(self.print_separater)
        print(self.config.userInterface.expID)
        
        accuracy_label0_realdata_list=list()
        accuracy_label0_fakedata_list=list()
        accuracy_label1_realdata_list=list()
        accuracy_label1_fakedata_list=list()
        contentFidList=list()
        styleFidList=list()
        
        # for feature extractors category
        for ii in range(len(inputIO.featureExtractorCategory.accuracy.realContent)): accuracy_label0_realdata_list.append(list())
        for ii in range(len(inputIO.featureExtractorCategory.accuracy.fakeContent)): accuracy_label0_fakedata_list.append(list())
        for ii in range(len(inputIO.featureExtractorCategory.accuracy.realStyle)): accuracy_label1_realdata_list.append(list())
        for ii in range(len(inputIO.featureExtractorCategory.accuracy.fakeStyle)): accuracy_label1_fakedata_list.append(list())
        for ii in range(len(inputIO.featureExtractorFid.content)): contentFidList.append(list())
        for ii in range(len(inputIO.featureExtractorFid.style)): styleFidList.append(list())
        
        # for generator category
        accuracy_label0_realdata_list.append(list())
        accuracy_label0_fakedata_list.append(list())
        accuracy_label1_realdata_list.append(list())
        accuracy_label1_fakedata_list.append(list())
        # contentFidList.append(list())
        # styleFidList.append(list())
        
        thisRoundStart=time.time()
        # iter_num=10
        for bid in range(iter_num):
            thisItrStart = time.time()
            # tmp=self.sess.run(inputIO.featureExtractorFid.content)
            if evalFID:
                evalOps = [inputIO.featureExtractorCategory.accuracy, inputIO.generatorCategory.accuracy, 
                               inputIO.featureExtractorFid.content, 
                               inputIO.featureExtractorFid.style]
            else:
                evalOps = [inputIO.featureExtractorCategory.accuracy, inputIO.generatorCategory.accuracy]
            
            evalResult = \
                self.sess.run(evalOps)
            
            if evalFID:
                featureExtractorAccuracy, generatorAccuracy, fidContent, fidStyle = evalResult
                for ii in range(len(fidContent)):
                    contentFidList[ii].append(fidContent[ii])
                for ii in range(len(fidStyle)):
                    styleFidList[ii].append(fidStyle[ii])
                
                
                
                # fidContent = np.average(fidContent)
                # fidStyle=np.average(fidStyle)
                # contentFidList.append(fidContent)
                # styleFidList.append(fidStyle)
            else:
                featureExtractorAccuracy, generatorAccuracy = evalResult
                for ii in range(len(inputIO.featureExtractorFid.content)):
                    contentFidList[ii].append(0)
                for ii in range(len(inputIO.featureExtractorFid.style)):
                    styleFidList[ii].append(0)
                
            for ii in range(len(featureExtractorAccuracy.realContent)): 
                accuracy_label0_realdata_list[ii].append(featureExtractorAccuracy.realContent[ii])
            for ii in range(len(featureExtractorAccuracy.fakeContent)): 
                accuracy_label0_fakedata_list[ii].append(featureExtractorAccuracy.fakeContent[ii])
            for ii in range(len(featureExtractorAccuracy.realStyle)): 
                accuracy_label1_realdata_list[ii].append(featureExtractorAccuracy.realStyle[ii])
            for ii in range(len(featureExtractorAccuracy.fakeStyle)): 
                accuracy_label1_fakedata_list[ii].append(featureExtractorAccuracy.fakeStyle[ii])
                
            # for ii in fidContet:
            
            
            
            accuracy_label0_realdata_list[-1].append(generatorAccuracy.contentReal)
            accuracy_label0_fakedata_list[-1].append(generatorAccuracy.contentFake)
            accuracy_label1_realdata_list[-1].append(generatorAccuracy.styleReal)
            accuracy_label1_fakedata_list[-1].append(generatorAccuracy.styleFake)
            
                
            print(progress_info + ": "+print_info + ", Iter %.2fs-%d/%d, CntR-%.1f%%, CntF-%.1f%%, CntFid-%.1f, StyR-%.1f%%, StyF-%.1f%%, StyFid-%0.1f" % 
                        (time.time()-thisItrStart, bid+1, iter_num, 
                         np.average(accuracy_label0_realdata_list[:-1]), 
                        np.average(accuracy_label0_fakedata_list[:-1]),
                        np.average(contentFidList),
                        np.average(accuracy_label1_realdata_list[:-1]),
                        np.average(accuracy_label1_fakedata_list[:-1]),
                        np.average(styleFidList)), end="\r")
                
            if bid ==0 or bid == iter_num-1 or time.time()-thisRoundStart>RECORD_TIME:
                    thisRoundStart = time.time()
                    print(progress_info + ": "+print_info + ", Iter %.2fs-%d/%d, CntR-%.1f%%, CntF-%.1f%%, CntFid-%.1f, StyR-%.1f%%, StyF-%.1f%%, StyFid-%0.1f;" % 
                        (time.time()-thisItrStart, bid+1, iter_num, 
                        np.average(accuracy_label0_realdata_list[:-1]), 
                        np.average(accuracy_label0_fakedata_list[:-1]),
                        np.average(contentFidList),
                        np.average(accuracy_label1_realdata_list[:-1]),
                        np.average(accuracy_label1_fakedata_list[:-1]),
                        np.average(styleFidList)))
                    
        for ii in range(len(accuracy_label0_realdata_list)-1):
            accuracy_label0_realdata_list[ii]=np.average(accuracy_label0_realdata_list[ii])
            _thisSummary=self.sess.run(summaryOp[0][0][ii], feed_dict={summaryOp[1][0][ii]:accuracy_label0_realdata_list[ii]})
            summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        for ii in range(len(accuracy_label0_fakedata_list)-1):
            accuracy_label0_fakedata_list[ii]=np.average(accuracy_label0_fakedata_list[ii])
            _thisSummary=self.sess.run(summaryOp[0][1][ii], feed_dict={summaryOp[1][1][ii]:accuracy_label0_fakedata_list[ii]})
            summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        for ii in range(len(accuracy_label1_realdata_list)-1):
            accuracy_label1_realdata_list[ii]=np.average(accuracy_label1_realdata_list[ii])
            _thisSummary=self.sess.run(summaryOp[0][2][ii], feed_dict={summaryOp[1][2][ii]:accuracy_label1_realdata_list[ii]})
            summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        for ii in range(len(accuracy_label1_fakedata_list)-1):
            accuracy_label1_fakedata_list[ii]=np.average(accuracy_label1_fakedata_list[ii])
            _thisSummary=self.sess.run(summaryOp[0][3][ii], feed_dict={summaryOp[1][3][ii]:accuracy_label1_fakedata_list[ii]})
            summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
            
        accuracy_label0_realdata_list[-1]=np.average(accuracy_label0_realdata_list[-1])
        _thisSummary=self.sess.run(summaryOp[0][0][-1], feed_dict={summaryOp[1][0][-1]:accuracy_label0_realdata_list[-1]})
        summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        
        accuracy_label0_fakedata_list[-1]=np.average(accuracy_label0_fakedata_list[-1])
        _thisSummary=self.sess.run(summaryOp[0][1][-1], feed_dict={summaryOp[1][1][-1]:accuracy_label0_fakedata_list[-1]})
        summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        
        accuracy_label1_realdata_list[-1]=np.average(accuracy_label1_realdata_list[-1])
        _thisSummary=self.sess.run(summaryOp[0][2][-1], feed_dict={summaryOp[1][2][-1]:accuracy_label1_realdata_list[-1]})
        summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        
        accuracy_label1_fakedata_list[-1]=np.average(accuracy_label1_fakedata_list[-1])
        _thisSummary=self.sess.run(summaryOp[0][3][-1], feed_dict={summaryOp[1][3][-1]:accuracy_label1_fakedata_list[-1]})
        summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
            
            
        ### fid summaries 
        if evalFID:
            for ii in range(len(contentFidList)):
                contentFidList[ii]=np.average(contentFidList[ii])
                _thisSummary=self.sess.run(summaryOp[0][4][ii], feed_dict={summaryOp[1][4][ii]:contentFidList[ii]})
                summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
            for ii in range(len(styleFidList)):
                styleFidList[ii]=np.average(styleFidList[ii])
                _thisSummary=self.sess.run(summaryOp[0][5][ii], feed_dict={summaryOp[1][5][ii]:contentFidList[ii]})
                summaryWriter.add_summary(_thisSummary, ei.eval(session=self.sess))
        
        
        
        summaryWriter.flush()
        print("Tensorboard Logs have been updated, elapsed %.2f secs." % (time.time()-thisRoundStart))
        return
                
                