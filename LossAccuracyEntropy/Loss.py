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
#HighLevelFeaturePenaltyPctg=[eps,eps,eps,0.25,0.3]
from Networks.NetworkClass import NetworkIO

import numpy as np


class InstanceLossIO(object):
    def __init__(self):
        self.sumLossG=0
        self.lossG=EasyDict({})
        self.sumLossFE=0
        self.lossFE=EasyDict({})

class Loss(object):
    def __init__(self, penalties):
        
        self.penalties=penalties
        self.IO=NetworkIO()
        return
    
    
    def BuildLosses(self, generatorIO, featureExtractorIO, validateOn='NA', isTrain=False):
        thisIO = InstanceLossIO()
        thisIO = self.GeneratorLoss(io=thisIO, generatorIO=generatorIO)
        if isTrain: 
            print("Generator Losses Created on %s" % validateOn)
        thisIO = self.FeatureExtractorLoss(io=thisIO, featureExtractorIO=featureExtractorIO)
        if isTrain: 
            print("Feature Extractor Losses Created on %s" % validateOn)
        return thisIO
        
    
    def GeneratorLoss(self, io, generatorIO):
        
        
        # Regularizations in the generator
        generator_weight_decay_loss = tf.get_collection('generator_weight_decay')
        weight_decay_loss = 0
        if self.penalties.generator_weight_decay_penalty>eps*10 and generator_weight_decay_loss:
            for ii in generator_weight_decay_loss:
                weight_decay_loss = ii + weight_decay_loss
            weight_decay_loss = weight_decay_loss / len(generator_weight_decay_loss)
            io.sumLossG+=weight_decay_loss
            io.lossG.regularization=weight_decay_loss/self.penalties.generator_weight_decay_penalty
            
        # L1 Loss
        if self.penalties.Pixel_Reconstruction_Penalty>eps*10:
            l1 = tf.abs(generatorIO.outputs.generated - generatorIO.groundtruths.trueCharacter)
            l1 = tf.reduce_mean(l1) * self.penalties.Pixel_Reconstruction_Penalty
            io.sumLossG+=l1
            io.lossG.L1=l1/self.penalties.Pixel_Reconstruction_Penalty
            
        # Const loss for content
        if self.penalties.Lconst_content_Penalty>eps*10:
            contentConstLoss = tf.square(generatorIO.outputs.constContent - generatorIO.outputs.encodedContentFeatures)
            contentConstLoss = tf.reduce_mean(contentConstLoss) * self.penalties.Lconst_content_Penalty
            io.sumLossG+=contentConstLoss
            io.lossG.ConstContent=contentConstLoss/self.penalties.Lconst_content_Penalty
        
        # Const loss for style
        if self.penalties.Lconst_style_Penalty > eps * 10:
            styleConstLoss=0
            for thisFullFeatures in generatorIO.outputs.constStyle:
                thisStyleConstLoss = tf.square(generatorIO.outputs.encodedStyleFeatures-thisFullFeatures)
                contentConstLoss = tf.reduce_mean(thisStyleConstLoss) 
                styleConstLoss+=thisStyleConstLoss
            styleConstLoss = tf.reduce_mean(styleConstLoss) * self.penalties.Lconst_style_Penalty
            io.sumLossG+=styleConstLoss
            io.lossG.ConstStyle=styleConstLoss/self.penalties.Lconst_style_Penalty
                
        
        
        # Categorical loss for the content && style encoders
        if self.penalties.Generator_Categorical_Penalty>10*eps:
            # for contents on the original input
            categoryContentOnOrg = tf.nn.softmax_cross_entropy_with_logits(logits=generatorIO.outputs.categoryLogitContentOrg,
                                                                           labels=generatorIO.groundtruths.onehotLabel0)
            categoryContentOnOrg = tf.reduce_mean(categoryContentOnOrg) * self.penalties.Generator_Categorical_Penalty
            
            # for contents on the generated 
            categoryContentOnGenerated = tf.nn.softmax_cross_entropy_with_logits(logits=generatorIO.outputs.categoryLogitContentGenerated,
                                                                                 labels=generatorIO.groundtruths.onehotLabel0)
            categoryContentOnGenerated = tf.reduce_mean(categoryContentOnGenerated) * self.penalties.Generator_Categorical_Penalty
            
            
            
            # for style on the original input
            categoryStyleOnOrg=0
            for logits in generatorIO.outputs.categoryLogitStyleOrg:
                thisCategoryStyleOnOrg = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                                 labels=generatorIO.groundtruths.onehotLabel1)
                categoryStyleOnOrg+=thisCategoryStyleOnOrg
            categoryStyleOnOrg = tf.reduce_mean(categoryStyleOnOrg) * self.penalties.Generator_Categorical_Penalty / len(generatorIO.outputs.categoryLogitStyleOrg)
            
            # for style on the generated 
            categoryStyleOnGenerated = tf.nn.softmax_cross_entropy_with_logits(logits=generatorIO.outputs.categoryLogitStyleGenerated,
                                                                               labels=generatorIO.groundtruths.onehotLabel1)
            categoryStyleOnGenerated = tf.reduce_mean(categoryStyleOnGenerated) * self.penalties.Generator_Categorical_Penalty
            
            io.sumLossG+=(categoryContentOnOrg+categoryContentOnGenerated+categoryStyleOnOrg+categoryContentOnGenerated)
            io.lossG.sumLossG = io.sumLossG
            io.lossG.CategoryContentOnOrg=categoryContentOnOrg/self.penalties.Generator_Categorical_Penalty
            io.lossG.CategoryContentOnGenerated=categoryContentOnGenerated/self.penalties.Generator_Categorical_Penalty
            io.lossG.CategoryStyleOnOrg=categoryStyleOnOrg/self.penalties.Generator_Categorical_Penalty
            io.lossG.CategoryStyleOnGenerated=categoryStyleOnGenerated/self.penalties.Generator_Categorical_Penalty
            
        return io
    
    def FeatureExtractorLoss(self, io, featureExtractorIO):
        
        def _featureLinearNorm(feature):
            min_v= tf.reduce_min(feature)
            feature = feature - min_v
            max_v = tf.reduce_max(feature)
            feature = feature / max_v
            return feature+eps
        
        def CalculateFID(feature1, feature2):
            
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
            #     covmean = tf.linalg.sqrtm(tf.matmul(cov1, cov2))
            #     covmean = tf.cast(tf.math.real(covmean), tf.float32)
            covmean = matrix_square_root(tf.matmul(cov1, cov2)) # as the approximation of matrix square root
            covmean = tf.where(tf.is_nan(covmean), tf.zeros_like(covmean), covmean)
            covmean = tf.where(tf.is_inf(covmean), tf.zeros_like(covmean), covmean)
            cov1 += tf.eye(tf.shape(cov1)[0]) * 1e-6
            cov2 += tf.eye(tf.shape(cov2)[0]) * 1e-6            
            fid = tf.reduce_sum(tf.square(diff)) + tf.linalg.trace(cov1 + cov2 - 2 * covmean)
            return fid
        
        
        def CalculateFeatureDifference(feature1,feature2):
            for counter in range(len(HighLevelFeaturePenaltyPctg)):
                feature_diff = feature1[counter] - feature2[counter]
                if not feature_diff.shape.ndims==4:
                    feature_diff = tf.reshape(feature_diff,[int(feature_diff.shape[0]),int(feature_diff.shape[1]),1,1])
                squared_feature_diff = feature_diff**2
                mean_squared_feature_diff = tf.reduce_mean(squared_feature_diff,axis=[1,2,3])
                square_root_mean_squared_feature_diff = tf.sqrt(eps+mean_squared_feature_diff)
                this_feature_loss = tf.reduce_mean(square_root_mean_squared_feature_diff)
                this_feature_loss = this_feature_loss * (HighLevelFeaturePenaltyPctg[counter]+eps)

                feature1_normed = _featureLinearNorm(feature=feature1[counter])
                feature2_normed = _featureLinearNorm(feature=feature2[counter])
                vn_loss = tf.trace(tf.multiply(feature1_normed, tf.log(feature1_normed)) -
                                   tf.multiply(feature1_normed, tf.log(feature2_normed)) +
                                   - feature1_normed + feature2_normed + eps)
                vn_loss = tf.reduce_mean(vn_loss)
                vn_loss = vn_loss * (HighLevelFeaturePenaltyPctg[counter]+eps)

                if counter == 0:
                    final_loss_mse = this_feature_loss
                    final_loss_vn = vn_loss 
                else:
                    final_loss_mse += this_feature_loss
                    final_loss_vn += vn_loss
            final_loss_mse = final_loss_mse / (sum(HighLevelFeaturePenaltyPctg)+eps)
            final_loss_vn = final_loss_vn / (sum(HighLevelFeaturePenaltyPctg)+eps)
            return final_loss_mse, final_loss_vn
        
        
        # penalties of content feature extractors
        if not len(self.penalties.FeatureExtractorPenalty_ContentPrototype)==\
                    len(featureExtractorIO.outputs.realContentFeatures)==\
                        len(featureExtractorIO.outputs.fakeContentFeatures):
            print("ERROR")
            return
        
        contentMSE=0
        contentVN=0
        mseContentList=[]
        vnContentList=[]
        fidContentList=[]
        for ii in range(len(self.penalties.FeatureExtractorPenalty_ContentPrototype)):
            _mse, _vn = CalculateFeatureDifference(feature1=featureExtractorIO.outputs.realContentFeatures[ii],
                                                   feature2=featureExtractorIO.outputs.fakeContentFeatures[ii],)
            _contentFid = CalculateFID(feature1=featureExtractorIO.outputs.realContentFidFeature[ii],
                                       feature2=featureExtractorIO.outputs.fakeContentFidFeature[ii])
            
            
            contentMSE+=_mse*(self.penalties.FeatureExtractorPenalty_ContentPrototype[ii]+eps)
            contentVN+=_vn*(self.penalties.FeatureExtractorPenalty_ContentPrototype[ii]+eps)
            mseContentList.append(_mse)
            vnContentList.append(_vn)
            fidContentList.append(_contentFid)
        
        
        # penalties of style feature extractors
        if not len(self.penalties.FeatureExtractorPenalty_StyleReference)==\
                    len(featureExtractorIO.outputs.realStyleFeatures)==\
                        len(featureExtractorIO.outputs.fakeStyleFeatures):
            print("ERROR")
            return
        
        styleMSE=0
        styleVN=0
        mseStyleList=[]
        vnStyleList=[]
        fidStyleList=[]
        for ii in range(len(self.penalties.FeatureExtractorPenalty_StyleReference)):
            _mse, _vn = CalculateFeatureDifference(feature1=featureExtractorIO.outputs.realStyleFeatures[ii],
                                                   feature2=featureExtractorIO.outputs.fakeStyleFeatures[ii])
            _styleFid = CalculateFID(feature1=featureExtractorIO.outputs.realStyleFidFeature[ii],
                                     feature2=featureExtractorIO.outputs.fakeStyleFidFeature[ii])
            
            styleMSE+=_mse*self.penalties.FeatureExtractorPenalty_StyleReference[ii]
            styleVN+=_vn*self.penalties.FeatureExtractorPenalty_StyleReference[ii]
            mseStyleList.append(_mse)
            vnStyleList.append(_vn)
            fidStyleList.append(_styleFid)
        
            
        #io.sumLossFE+=contentMSE+contentVN+styleMSE+styleVN
        io.sumLossFE+=contentMSE+styleMSE
        io.lossFE.content = (contentMSE+contentVN)/(np.sum(self.penalties.FeatureExtractorPenalty_ContentPrototype)+eps)
        io.lossFE.style = (styleMSE+styleVN)/(np.sum(self.penalties.FeatureExtractorPenalty_StyleReference)+eps)
        io.lossFE.mseContent=mseContentList 
        io.lossFE.vnContent=vnContentList
        io.lossFE.mseStyle=mseStyleList
        io.lossFE.vnStyle=vnStyleList
        
        io.lossFE.fidContent=fidContentList
        io.lossFE.fidStyle=fidStyleList
        io.lossFE.fidContentSum=tf.reduce_mean(fidContentList)
        io.lossFE.fidStyleSum=tf.reduce_mean(fidStyleList)
        return io
        
