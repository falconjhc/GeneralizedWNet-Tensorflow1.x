# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../')
sys.path.append('../../')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from easydict import EasyDict
eps = 1e-9

HighLevelFeaturePenaltyPctg=[0.1,0.15,0.2,0.25,0.3]
from Networks.NetworkClass import NetworkIO

class InstanceLossIO(object):
    def __init__(self):
        
        self.generatorCategory = EasyDict({})
        self.featureExtractorCategory = EasyDict({})
        self.featureExtractorFid=EasyDict({})

        
        self.generatorCategory.accuracy=EasyDict({})
        self.generatorCategory.entropy=EasyDict({})
        
        
        self.featureExtractorCategory.accuracy=EasyDict({})
        self.featureExtractorCategory.entropy=EasyDict({})


class AccuracyAndEntropy(object):
    def __init__(self):
        self.IO=NetworkIO()

    
    def CalculateAccuracyAndEntropy(self, logits, gtLabels):
        prdt_labels = tf.argmax(logits,axis=1)
        true_labels = tf.argmax(gtLabels,axis=1)
        correct_prediction = tf.equal(prdt_labels,true_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) * 100
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.nn.softmax(logits))
        entropy = tf.reduce_mean(entropy)
        return accuracy,entropy
    
    def CalculateFID(self, feature1, feature2):
        def matrix_square_root(A, num_iters=100):
                dim = tf.shape(A)[0]
                normA = tf.reduce_sum(A * A)
                Y = A / normA
                I = tf.eye(dim)
                Z = I
                for _ in range(num_iters):
                    T = 0.5 * (3.0 * I - tf.matmul(Z, Y))
                    Y = tf.matmul(Y, T)
                    Z = tf.matmul(T, Z)
                sqrtA = Y * tf.sqrt(normA)
                return sqrtA

    
        # with tf.device(device):
        feature1=tf.reshape(feature1, (feature1.shape[0],-1))
        feature2=tf.reshape(feature2, (feature2.shape[0],-1))
        mean1 = tf.reduce_mean(feature1, axis=0)
        mean2 = tf.reduce_mean(feature2, axis=0)
        cov1 = tf.matmul(tf.transpose(feature1 - mean1), (feature1 - mean1)) / (tf.cast(tf.shape(feature1)[0], tf.float32) - 1)
        cov2 = tf.matmul(tf.transpose(feature2 - mean2), (feature2 - mean2)) / (tf.cast(tf.shape(feature2)[0], tf.float32) - 1)
        diff = mean1 - mean2
        # covmean = tf.linalg.sqrtm(tf.matmul(cov1, cov2))
        # covmean = tf.cast(tf.math.real(covmean), tf.float32)
        covmean = matrix_square_root(tf.matmul(cov1, cov2)) # as the approximation of matrix square root
        covmean = tf.where(tf.is_nan(covmean), tf.zeros_like(covmean), covmean)
        covmean = tf.where(tf.is_inf(covmean), tf.zeros_like(covmean), covmean)
        cov1 += tf.eye(tf.shape(cov1)[0]) * 1e-6
        cov2 += tf.eye(tf.shape(cov2)[0]) * 1e-6            
        fid = tf.reduce_sum(tf.square(diff)) + tf.linalg.trace(cov1 + cov2 - 2 * covmean)
        return fid
        
        
    
    def BuildAccuracy(self, generatorIO, featureExtractorIO, validateOn='NA'):
        thisIO=InstanceLossIO()
        
        groundtruthLabel0=featureExtractorIO.groundtruths.onehotLabel0
        groundtruthLabel1=featureExtractorIO.groundtruths.onehotLabel1
        
        
        
        
        # Generator cagetories
        thisIO.generatorCategory.accuracy.contentReal, thisIO.generatorCategory.entropy.contentReal=\
            self.CalculateAccuracyAndEntropy(logits=generatorIO.outputs.categoryLogitContentOrg[0],
                                             gtLabels=groundtruthLabel0)
        
        thisIO.generatorCategory.accuracy.contentFake, thisIO.generatorCategory.entropy.contentFake=\
            self.CalculateAccuracyAndEntropy(logits=generatorIO.outputs.categoryLogitContentGenerated[0],
                                             gtLabels=groundtruthLabel0)
        
        # styleOrgAccuracy=[]
        # styleOrgEntropy=[]
        for ii in range(len(generatorIO.outputs.categoryLogitStyleOrg)):
            thisAccuracy, thisEntropy=\
                self.CalculateAccuracyAndEntropy(logits=generatorIO.outputs.categoryLogitStyleOrg[ii][0],
                                                 gtLabels=groundtruthLabel1)
                
            if ii ==0:
                thisIO.generatorCategory.accuracy.styleReal = tf.expand_dims(tf.expand_dims(thisAccuracy, axis=0),axis=0)
                thisIO.generatorCategory.entropy.styleReal = tf.expand_dims(tf.expand_dims(thisEntropy, axis=0), axis=0)
            else:
                thisIO.generatorCategory.accuracy.styleReal = tf.concat([thisIO.generatorCategory.accuracy.styleReal,  tf.expand_dims(tf.expand_dims(thisAccuracy, axis=0),axis=0)], axis=0)
                thisIO.generatorCategory.entropy.styleReal = tf.concat([thisIO.generatorCategory.entropy.styleReal, tf.expand_dims(tf.expand_dims(thisEntropy, axis=0), axis=0)], axis=0)
                
        thisIO.generatorCategory.accuracy.styleReal = tf.reduce_mean(thisIO.generatorCategory.accuracy.styleReal)
        thisIO.generatorCategory.entropy.styleReal = tf.reduce_mean(thisIO.generatorCategory.entropy.styleReal)
            
        thisIO.generatorCategory.accuracy.styleFake, thisIO.generatorCategory.entropy.styleFake=\
            self.CalculateAccuracyAndEntropy(logits=generatorIO.outputs.categoryLogitStyleGenerated[0],
                                             gtLabels=groundtruthLabel1)
        
        
        # Feature extractor fids
        thisIO.featureExtractorFid.content=list()
        for ii in range(len(featureExtractorIO.outputs.realContentLogits)):
            thisFid=self.CalculateFID(feature1=featureExtractorIO.outputs.realContentFidFeature[ii], 
                                      feature2=featureExtractorIO.outputs.fakeContentFidFeature[ii])
            thisIO.featureExtractorFid.content.append(thisFid)
        
        thisIO.featureExtractorFid.style=list()
        for ii in range(len(featureExtractorIO.outputs.realContentLogits)):
            thisFid=self.CalculateFID(feature1=featureExtractorIO.outputs.realStyleFidFeature[ii], 
                                      feature2=featureExtractorIO.outputs.fakeStyleFidFeature[ii])
            thisIO.featureExtractorFid.style.append(thisFid)
        
        
        # feature extractor categories
        thisIO.featureExtractorCategory.accuracy.realContent=list()
        thisIO.featureExtractorCategory.entropy.realContent=list()
        for ii in range(len(featureExtractorIO.outputs.realContentLogits)):
            thisContentAccuracy, thisContentEntropy =\
                self.CalculateAccuracyAndEntropy(logits=featureExtractorIO.outputs.realContentLogits[ii],
                                                 gtLabels=groundtruthLabel0)
            thisIO.featureExtractorCategory.accuracy.realContent.append(thisContentAccuracy)
            thisIO.featureExtractorCategory.entropy.realContent.append(thisContentEntropy)
                
                
        thisIO.featureExtractorCategory.accuracy.fakeContent=list()
        thisIO.featureExtractorCategory.entropy.fakeContent=list()
        for ii in range(len(featureExtractorIO.outputs.fakeContentLogits)):
            thisContentAccuracy, thisContentEntropy =\
                self.CalculateAccuracyAndEntropy(logits=featureExtractorIO.outputs.fakeContentLogits[ii],
                                                 gtLabels=groundtruthLabel0)
            thisIO.featureExtractorCategory.accuracy.fakeContent.append(thisContentAccuracy)
            thisIO.featureExtractorCategory.entropy.fakeContent.append(thisContentEntropy)
                
        thisIO.featureExtractorCategory.accuracy.realStyle=list()
        thisIO.featureExtractorCategory.entropy.realStyle=list()
        for ii in range(len(featureExtractorIO.outputs.realStyleLogits)):
            thisStyleAccuracy, thisStyleEntropy =\
                self.CalculateAccuracyAndEntropy(logits=featureExtractorIO.outputs.realStyleLogits[ii],
                                                 gtLabels=groundtruthLabel1)
            thisIO.featureExtractorCategory.accuracy.realStyle.append(thisStyleAccuracy)
            thisIO.featureExtractorCategory.entropy.realStyle.append(thisStyleEntropy)
            
        thisIO.featureExtractorCategory.accuracy.fakeStyle=list()
        thisIO.featureExtractorCategory.entropy.fakeStyle=list()
        for ii in range(len(featureExtractorIO.outputs.fakeStyleLogits)):
            thisStyleAccuracy, thisStyleEntropy =\
                self.CalculateAccuracyAndEntropy(logits=featureExtractorIO.outputs.fakeStyleLogits[ii],
                                                 gtLabels=groundtruthLabel1)
            thisIO.featureExtractorCategory.accuracy.fakeStyle.append(thisStyleAccuracy)
            thisIO.featureExtractorCategory.entropy.fakeStyle.append(thisStyleEntropy)
    
        print("Accuracy created on %s" % validateOn)
        return thisIO
                
    