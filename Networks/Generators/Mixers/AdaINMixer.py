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


eps = 1e-9
generator_dim = 64


residualAtLayer=3
residualBlockNum=5
adain=0

print_separater="#########################################################"

def _calculate_batch_diff(input_feature):
    if len(input_feature.shape)==3:
        diff = tf.abs(tf.expand_dims(input_feature, 3) -
                      tf.expand_dims(tf.transpose(input_feature, [1, 2, 0]), 0))
        diff = tf.reduce_sum(tf.exp(-diff), 3)
    elif len(input_feature.shape)==4:
        diff = tf.abs(tf.expand_dims(input_feature, 4) -
                      tf.expand_dims(tf.transpose(input_feature, [1, 2, 3, 0]), 0))
        diff = tf.reduce_sum(tf.exp(-diff), 4)
    return tf.reduce_mean(diff)


class WNetMixer(NetworkBase):
    def __init__(self, inputFromContentEncoder, inputFromStyleEncoder, 
                 config, scope, initializer, penalties):
        
        super().__init__()
        
        
        self.config = config
        self.scope=scope
        self.initializer = initializer
        self.penalties = penalties
        
        self.inputs.update({'fromContentEncoder': inputFromContentEncoder})
        self.inputs.update({'fromStyleEncoder': inputFromStyleEncoder})
        self.outputs.update({'fusedFeatures': None})
        self.outputs.update({'encodedStyleFinalOutput': None})
        self.outputs.update({'styleShortcutBatchDiff': None})
        self.outputs.update({'styleResidualBatchDiff': None})
        
        return
    
    def FuseFeature(self, reuse):
        # multiple encoded information average calculation for style reference encoder
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.config.generator.device):
                with tf.variable_scope(self.scope):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    for ii in range(self.config.datasetConfig.inputStyleNum):
                        if ii == 0:
                            encoded_style_final = tf.expand_dims(self.inputs.fromStyleEncoder.encodedFinalOutputList[ii][0], axis=0)
                            style_short_cut_interface = list()
                            if len(self.inputs.fromStyleEncoder.shortcutOutputList)>0:
                                for jj in range(len(self.inputs.fromStyleEncoder.shortcutOutputList[ii])):
                                    style_short_cut_interface.append(
                                        tf.expand_dims(self.inputs.fromStyleEncoder.shortcutOutputList[ii][jj], axis=0))
                            style_residual_interface = list()
                            if len(self.inputs.fromStyleEncoder.residualOutputList)>0:
                                for jj in range(len(self.inputs.fromStyleEncoder.residualOutputList[ii])):
                                    style_residual_interface.append(
                                        tf.expand_dims(self.inputs.fromStyleEncoder.residualOutputList[ii][jj], axis=0))
                        else:
                            encoded_style_final = tf.concat([encoded_style_final, tf.expand_dims(self.inputs.fromStyleEncoder.encodedFinalOutputList[ii][0], axis=0)], axis=0)
                            for jj in range(len(self.inputs.fromStyleEncoder.shortcutOutputList[ii])):
                                style_short_cut_interface[jj] = tf.concat([style_short_cut_interface[jj],
                                                                        tf.expand_dims(self.inputs.fromStyleEncoder.shortcutOutputList[ii][jj],
                                                                                       axis=0)], axis=0)
                            if len(self.inputs.fromStyleEncoder.residualOutputList)>0:
                                for jj in range(len(self.inputs.fromStyleEncoder.residualOutputList[ii])):
                                    style_residual_interface[jj] = tf.concat([style_residual_interface[jj], 
                                                                            tf.expand_dims(self.inputs.fromStyleEncoder.residualOutputList[ii][jj], 
                                                                                            axis=0)], axis=0)

                    # encoded_style_final_avg = tf.reduce_mean(encoded_style_final, axis=0)
                    encoded_style_final_max = tf.reduce_max(encoded_style_final, axis=0)
                    # encoded_style_final_min = tf.reduce_min(encoded_style_final, axis=0)
                    # encoded_style_final = tf.concat([encoded_style_final_avg, encoded_style_final_max, encoded_style_final_min], axis=3)
                    # encoded_style_final = tf.concat([encoded_style_final_avg, encoded_style_final_max], axis=3)
                    encoded_style_final = encoded_style_final_max

                    style_shortcut_batch_diff = 0
                    for ii in range(len(style_short_cut_interface)):
                        #style_short_cut_avg = tf.reduce_mean(style_short_cut_interface[ii], axis=0)
                        style_short_cut_max = tf.reduce_max(style_short_cut_interface[ii], axis=0)
                        #style_short_cut_min = tf.reduce_min(style_short_cut_interface[ii], axis=0)
                        #style_short_cut_interface[ii] = tf.concat([style_short_cut_avg, style_short_cut_max, style_short_cut_min], axis=3)
                        style_short_cut_interface[ii] = style_short_cut_max
                        style_shortcut_batch_diff += _calculate_batch_diff(input_feature=style_short_cut_interface[ii])
                    style_shortcut_batch_diff = style_shortcut_batch_diff / (len(style_short_cut_interface)+eps)

                    style_residual_batch_diff = 0
                    for ii in range(len(style_residual_interface)):
                        #style_residual_avg = tf.reduce_mean(style_residual_interface[ii], axis=0)
                        style_residual_max = tf.reduce_max(style_residual_interface[ii], axis=0)
                        #style_residual_min = tf.reduce_min(style_residual_interface[ii], axis=0)
                        #style_residual_interface[ii] = tf.concat([style_residual_avg, style_residual_max, style_residual_min], axis=3)
                        style_residual_interface[ii] = style_residual_max
                        style_residual_batch_diff += _calculate_batch_diff(input_feature=style_residual_interface[ii])
                    style_residual_batch_diff = style_residual_batch_diff / (len(style_residual_interface)+eps)

        # full style feature reformat
        if adain:
            full_style_feature_list_reformat = list()
            for ii in range(len(self.inputs.fromStyleEncoder.fullFeatureList)):
                for jj in range(len(self.inputs.fromStyleEncoder.fullFeatureList[ii])):
                    current_feature = tf.expand_dims(self.inputs.fromStyleEncoder.fullFeatureList[ii][jj], axis=0)
                    if ii == 0:
                        full_style_feature_list_reformat.append(current_feature)
                    else:
                        full_style_feature_list_reformat[jj] = tf.concat(
                            [full_style_feature_list_reformat[jj], current_feature], axis=0)
        else:
            full_style_feature_list_reformat = None

        # residual interfaces && short cut interfaces are fused together
        fused_residual_interfaces = list()
        fused_shortcut_interfaces = list()
        for ii in range(len(self.inputs.fromContentEncoder.residualOutputList)):
            current_content_residual_size = int(self.inputs.fromContentEncoder.residualOutputList[ii].shape[1])
            output_current_residual = self.inputs.fromContentEncoder.residualOutputList[ii]
            if adain:  # for adaptive instance normalization
                for jj in range(len(full_style_feature_list_reformat)):
                    if int(full_style_feature_list_reformat[jj].shape[2]) == int(output_current_residual.shape[1]):
                        break
                output_current_residual = adaptive_instance_norm(content=output_current_residual,
                                                                style=full_style_feature_list_reformat[jj])

            for jj in range(len(style_residual_interface)):
                current_style_residual_size = int(style_residual_interface[jj].shape[1])
                if current_style_residual_size == current_content_residual_size:
                    output_current_residual = tf.concat([output_current_residual, style_residual_interface[jj]], axis=3)
            fused_residual_interfaces.append(output_current_residual)
        for ii in range(len(self.inputs.fromContentEncoder.shortcutOutputList)):
            # current_content_shortcut_size = int(self.inputs.fromContentEncoder.shortcutOutputList[ii].shape[1])
            # output_current_shortcut = self.inputs.fromContentEncoder.shortcutOutputList[ii]
            # if adain:  # for adaptive instance normalization
            #     for jj in range(len(full_style_feature_list_reformat)):
            #         if int(full_style_feature_list_reformat[jj].shape[2]) == int(output_current_shortcut.shape[1]):
            #             break
            #     output_current_shortcut = adaptive_instance_norm(content=output_current_shortcut,
            #                                                     style=full_style_feature_list_reformat[jj])
            
            output_current_shortcut = tf.concat([self.inputs.fromContentEncoder.shortcutOutputList[ii], 
                                                 style_short_cut_interface[ii]],axis=-1)
            
            # for jj in range(len(style_short_cut_interface)):
            #     current_style_short_cut_size = int(style_short_cut_interface[jj].shape[1])
            #     if current_style_short_cut_size == current_content_shortcut_size:
            #         output_current_shortcut = tf.concat([output_current_shortcut, style_short_cut_interface[jj]],axis=-1)
            fused_shortcut_interfaces.append(output_current_shortcut)
            
        return fused_residual_interfaces, fused_shortcut_interfaces, full_style_feature_list_reformat, encoded_style_final, style_shortcut_batch_diff, style_residual_batch_diff
            
    def resBlock(self,
                 is_training, scope,
                 x,layer,style, filters):

        if 'SimpleMixer' in self.config.generator.mixer:
            output = x
            return output
        
        if not adain:
            norm1 = batch_norm(x=x,
                            is_training=is_training,
                            scope="Layer%dBN1" % layer,
                            parameter_update_device=self.config.generator.device)
        else:
            if self.config.generator.mixer == 'DenseMixer':
                travel_times = int(int(x.shape[3]) / int(style.shape[4]))
                style_tile = tf.tile(style,[1,1,1,1,travel_times])
                norm1 = adaptive_instance_norm(content=x,
                                            style=style_tile)
            elif self.config.generator.mixer == 'ResidualMixer':
                norm1 = adaptive_instance_norm(content=x,
                                            style=style)

        act1 = relu(norm1)
        conv1 = conv2d(x=act1,
                    output_filters=filters,
                    scope="Layer%dConv1" % layer,
                    parameter_update_device=self.config.generator.device,
                    kh=3,kw=3,sh=1,sw=1,
                    initializer=self.initializer,
                    weight_decay=self.penalties.generator_weight_decay_penalty,
                    name_prefix=scope)
        if not adain:
            norm2 = batch_norm(x=conv1,
                            is_training=is_training,
                            scope="Layer%dBN2" % layer,
                            parameter_update_device=self.config.generator.device)
        else:

            norm2 = adaptive_instance_norm(content=conv1,
                                        style=style)
        act2 = relu(norm2)
        conv2 = conv2d(x=act2,
                    output_filters=filters,
                    scope="Layer%dConv2" % layer,
                    parameter_update_device=self.config.generator.device,
                    initializer=self.initializer,
                    weight_decay=self.penalties.generator_weight_decay_penalty,
                    name_prefix=scope,
                    kh=3,kw=3,sh=1,sw=1)

        if 'ResidualMixer' in self.config.generator.mixer:
            output = x + conv2
        elif 'DenseMixer' in self.config.generator.mixer:
            output = conv2


        return output

    def ProcessFusedFeature(self,reuse,is_training, 
                            input_list,style_features,
                            scope):
        
        residual_connection_mode=None
        input_list.reverse()
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.config.generator.device):
                residual_output_list = list()

                if is_training  and not reuse:
                    print(print_separater)
                    print(self.scope + " Residual %d Blocks" % residualBlockNum)
                    print(self.scope + ' Adaptive Instance Normalization for Residual Preparations: %s' % residual_connection_mode)

                for ii in range(len(input_list)):
                    current_residual_num = residualBlockNum + 2 * ii
                    current_residual_input = input_list[ii]
                    current_scope = scope + '/onEncDecLyr%d' % (residualAtLayer - ii)


                    if adain:
                        with tf.variable_scope(current_scope):
                            for jj in range(len(style_features)):
                                if int(style_features[jj].shape[2]) == int(current_residual_input.shape[1]):
                                    break

                            for jj in range(int(style_features[ii].shape[0])):
                                if reuse or jj > 0:
                                    tf.get_variable_scope().reuse_variables()

                                batch_size = int(style_features[ii][jj, :, :, :, :].shape[0])
                                if batch_size == 1:
                                    current_init_residual_input = style_features[ii][jj, :, :, :, :]
                                else:
                                    current_init_residual_input = tf.squeeze(style_features[ii][jj, :, :, :, :])

                                if residual_connection_mode == 'Multi':
                                    # multiple cnn layer built to make the style_conv be incorporated with the dimension of the residual blocks
                                    log_input = math.log(int(current_init_residual_input.shape[3])) / math.log(2)
                                    if math.log(int(current_init_residual_input.shape[3])) < math.log(int(current_residual_input.shape[3])):
                                        if np.floor(log_input) < math.log(int(current_residual_input.shape[3])) / math.log(2):
                                            filter_num_start = int(np.floor(log_input)) + 1
                                        else:
                                            filter_num_start = int(np.floor(log_input))
                                        filter_num_start = int(math.pow(2,filter_num_start))
                                    elif math.log(int(current_init_residual_input.shape[3])) > math.log(int(current_residual_input.shape[3])):
                                        if np.ceil(log_input) > math.log(int(current_residual_input.shape[3])) / math.log(2):
                                            filter_num_start = int(np.ceil(log_input)) - 1
                                        else:
                                            filter_num_start = int(np.ceil(log_input))
                                        filter_num_start = int(math.pow(2, filter_num_start))
                                    else:
                                        filter_num_start = int(current_residual_input.shape[3])
                                    filter_num_end = int(current_residual_input.shape[3])

                                    if int(current_init_residual_input.shape[3]) == filter_num_end:
                                        continue_build = False
                                        style_conv = current_init_residual_input
                                    else:
                                        continue_build = True


                                    current_style_conv_input = current_init_residual_input
                                    current_output_filter_num = filter_num_start
                                    style_cnn_layer_num = 0
                                    while continue_build:
                                        style_conv = conv2d(x=current_style_conv_input,
                                                            output_filters=current_output_filter_num,
                                                            scope="Conv0StyleLayer%d" % (style_cnn_layer_num+1),
                                                            parameter_update_device=self.config.generator.device,
                                                            kh=3, kw=3, sh=1, sw=1,
                                                            initializer=self.initializer,
                                                            weight_decay=self.penalties.generator_weight_decay_penalty,
                                                            name_prefix=scope)
                                        if not (reuse or jj > 0):
                                            print (style_conv)
                                        style_conv = relu(style_conv)


                                        current_style_conv_input = style_conv

                                        if filter_num_start < filter_num_end:
                                            current_output_filter_num = current_output_filter_num * 2
                                        else:
                                            current_output_filter_num = int(current_output_filter_num / 2)
                                        style_cnn_layer_num += 1

                                        if current_output_filter_num > filter_num_end and \
                                                math.log(int(current_init_residual_input.shape[3])) \
                                                < math.log(int(current_residual_input.shape[3])):
                                            current_output_filter_num = filter_num_end
                                        if current_output_filter_num < filter_num_end and \
                                                math.log(int(current_init_residual_input.shape[3])) \
                                                > math.log(int(current_residual_input.shape[3])):
                                            current_output_filter_num = filter_num_end

                                        if int(style_conv.shape[3]) == filter_num_end:
                                            continue_build = False

                                elif residual_connection_mode == 'Single':
                                    if int(current_init_residual_input.shape[3]) == int(current_residual_input.shape[3]):
                                        style_conv = current_init_residual_input
                                    else:
                                        style_conv = conv2d(x=current_init_residual_input,
                                                            output_filters=int(current_residual_input.shape[3]),
                                                            scope="Conv0StyleLayer0",
                                                            parameter_update_device=self.config.generator.device,
                                                            kh=3, kw=3, sh=1, sw=1,
                                                            initializer=self.initializer,
                                                            weight_decay=self.penalties.generator_weight_decay_penalty,
                                                            name_prefix=scope)
                                        if not (reuse or jj > 0):
                                            print (style_conv)
                                        style_conv = relu(style_conv)

                                if jj == 0:
                                    style_features_new = tf.expand_dims(style_conv, axis=0)
                                else:
                                    style_features_new = tf.concat([style_features_new,
                                                                    tf.expand_dims(style_conv, axis=0)],
                                                                axis=0)

                        if (not reuse) and (not math.log(int(current_init_residual_input.shape[3])) == math.log(int(current_residual_input.shape[3]))):
                            print (print_separater)
                    else:
                        style_features_new=None


                    with tf.variable_scope(current_scope):
                        if reuse:
                            tf.get_variable_scope().reuse_variables()

                        tmp_residual_output_list_on_current_place = list()
                        filter_num = int(current_residual_input.shape[3])
                        for jj in range(current_residual_num):
                            if jj == 0:
                                residual_input = current_residual_input
                            else:
                                if self.config.generator.mixer == 'DenseMixer':
                                    for kk in range(len(tmp_residual_output_list_on_current_place)):
                                        if kk == 0:
                                            residual_input = tmp_residual_output_list_on_current_place[kk]
                                        else:
                                            residual_input = tf.concat([residual_input,
                                                                        tmp_residual_output_list_on_current_place[kk]],
                                                                    axis=3)
                                elif self.config.generator.mixer=='ResidualMixer':
                                    residual_input = residual_block_output  # noqa: F821
                                    
                            residual_block_output = \
                                self.resBlock(is_training=is_training,
                                                scope=scope,
                                                x=residual_input,
                                                layer=jj+1,
                                                style=style_features_new,
                                                filters=filter_num)
                            tmp_residual_output_list_on_current_place.append(residual_block_output)
                            if jj == current_residual_num-1:
                                residual_output = residual_block_output

                        residual_output_list.append(residual_output)


        # if (not reuse) and adain_use and (not debug_mode):
        #     print(print_separater)
        #     raw_input("Press enter to continue")
        # print(print_separater)

        return residual_output_list

    
    
    def BuildMixer(self,
                   is_training, residual_connection_mode,
                   reuse = False, saveEpochs=-1):
        
        
        
        fused_residual_interfaces,          \
        fused_shortcut_interfaces,          \
        full_style_feature_list_reformat,   \
        encodedStyleFinalOutput,\
        styleShortcutBatchDiff, styleResidualBatchDiff = self.FuseFeature(reuse=reuse)
        

        # fused resudual interfaces are put into the residual blocks
        if not residualBlockNum == 0 or not residualAtLayer == -1:
            residual_output_list = self.ProcessFusedFeature(reuse=reuse,is_training=is_training,
                                                               input_list=fused_residual_interfaces,
                                                               scope=self.scope + '-'+ self.config.generator.mixer,
                                                               style_features=full_style_feature_list_reformat)

        # combination of all the encoder outputs
        fused_shortcut_interfaces.reverse()
        fusedFeatures = fused_shortcut_interfaces + residual_output_list
        
        
        self.outputs.fusedFeatures=fusedFeatures
        self.outputs.encodedStyleFinalOutput=encodedStyleFinalOutput
        self.outputs.styleShortcutBatchDiff=styleShortcutBatchDiff
        self.outputs.styleResidualBatchDiff=styleResidualBatchDiff
        
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
            print(print_separater)
        return
        
            