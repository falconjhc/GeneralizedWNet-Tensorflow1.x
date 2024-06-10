from Networks.NetworkClass import NetworkBase
import sys
sys.path.append('../')
sys.path.append('../../')


import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()


import sys
sys.path.append('..')
cnnDim=64
vitDim=96
from Utilities.Blocks import patchSize

import numpy as np
import math
from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock
from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
from Utilities.utils import PrintNetworkVars
from Utilities.Blocks import DecodingBasicBlock as BasicBlock
from Utilities.Blocks import DecodingBottleneckBlock as BottleneckBlock
from Utilities.Blocks import DecodingVisionTransformerBlock as VisionTransformerBlock
from prettytable import PrettyTable
from Utilities.utils import SplitName

BlockDict={'Cv': BasicBlock,
           'Cbb': BasicBlock,
           'Cbn': BottleneckBlock,
           'Vit': VisionTransformerBlock}

def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)

print_separater="#########################################################"


class GeneralizedDecoder(NetworkBase):
    def __init__(self, inputFromContentEncoder, inputFromStyleEncoder, inputFromMixer,
                 config, scope, initializer, penalties):
        
        super().__init__()
        
        self.inputs.update({'fromContentEncoder': inputFromContentEncoder})
        self.inputs.update({'fromStyleEncoder': inputFromStyleEncoder})
        self.inputs.update({'fromMixer': inputFromMixer})
        
        self.outputs.update({'generated': None})
        self.outputs.update({'fullFeatureList': None})
        
        
        self.config = config
        self.scope=scope
        self.initializer = initializer
        self.penalties = penalties

        
        return
    
    
    def BuildDecoder(self, 
                     is_training,
                     reuse = False, saveEpochs=-1):
        architectureList = SplitName(self.config.generator.decoder)[1:]
        
        self.outputs.fullFeatureList=list()
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device(self.config.generator.device):
                with tf.variable_scope(self.scope):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    
                    # Block 1
                    # input 8x8xdim*16 / 4xdim*16, short 8x8xDIM / 4xDIM, output 16x16xdim*8 / 16xdim*8
                    downVitDim = (self.config.datasetConfig.imgWidth // patchSize )**2
                    thisBlock = FindKeys(BlockDict, architectureList[0])[0]
                    result1 = \
                        thisBlock(x=self.inputs.fromMixer.encodedFinalOutput, 
                                  dims={'MapC': cnnDim*8, 'VitC': vitDim*8,
                                        'HW': self.config.datasetConfig.imgWidth//8,'VitDim': downVitDim//64}, 
                                  config={'option': architectureList[0]},
                                  blockCount=1, 
                                  device=self.config.generator.device,
                                  weightDecay=self.penalties.generator_weight_decay_penalty,
                                  initializer=self.initializer,
                                  isTraining=is_training,
                                  encLayer=self.inputs.fromMixer.fusedFeatures[-1])
                    self.outputs.fullFeatureList.append(result1)
                    
                    # Block 2
                    # input 16x16xdim*8 / 16xdim*16, short 16x16xDIM / 16xDIM, output 32x32xdim*4 / 64xdim*4
                    thisBlock = FindKeys(BlockDict, architectureList[1])[0]
                    result2 = \
                        thisBlock(x=result1, 
                                  dims={'MapC': cnnDim*4, 'VitC': vitDim*4,
                                        'HW': self.config.datasetConfig.imgWidth//4,'VitDim': downVitDim//16}, 
                                  config={'option': architectureList[1]},
                                  blockCount=2, 
                                  device=self.config.generator.device,
                                  weightDecay=self.penalties.generator_weight_decay_penalty,
                                  initializer=self.initializer,
                                  isTraining=is_training,
                                  encLayer=self.inputs.fromMixer.fusedFeatures[-2])
                    self.outputs.fullFeatureList.append(result2)
                    
                    # Block 3
                    # input 32x32xdim*4 / 64xdim*4, short 32x32xDIM / 64xDIM, output 64x64xdim*2 / 256xdim*2
                    thisBlock = FindKeys(BlockDict, architectureList[2])[0]
                    result3 =\
                        thisBlock(x=result2, 
                                  dims={'MapC': cnnDim*2, 'VitC': vitDim*2,
                                        'HW': self.config.datasetConfig.imgWidth//2,'VitDim': downVitDim//4}, 
                                  config={'option': architectureList[2]},
                                  blockCount=3, 
                                  device=self.config.generator.device,
                                  weightDecay=self.penalties.generator_weight_decay_penalty,
                                  initializer=self.initializer,
                                  isTraining=is_training,
                                  encLayer=self.inputs.fromMixer.fusedFeatures[-3])
                    self.outputs.fullFeatureList.append(result3)
                    
                    # Block 4
                    # input 64x64xdim*2 (no upsample needed) / 256xdim*2 (no upsample needed), short 64x64xDIM / 256xDIM, 
                    # output 64x64xdim*1 / 256xdim*1
                    thisBlock = FindKeys(BlockDict, architectureList[3])[0]
                    result4 = \
                        thisBlock(x=result3, 
                                  dims={'MapC': cnnDim, 'VitC': vitDim,
                                        'HW': self.config.datasetConfig.imgWidth,'VitDim': downVitDim}, 
                                  config={'option': architectureList[3]},
                                  blockCount=4, 
                                  device=self.config.generator.device,
                                  weightDecay=self.penalties.generator_weight_decay_penalty,
                                  initializer=self.initializer,
                                  isTraining=is_training,
                                  encLayer=self.inputs.fromMixer.fusedFeatures[-4])
                    self.outputs.fullFeatureList.append(result4)
                    
                    # Block Last:
                    result5 = \
                        thisBlock(x=result4, 
                                  dims={'MapC': 1, 'VitC': patchSize*patchSize,
                                        'HW': self.config.datasetConfig.imgWidth,'VitDim': downVitDim}, 
                                  config={'option': architectureList[3]},
                                  blockCount=5, 
                                  device=self.config.generator.device,
                                  weightDecay=self.penalties.generator_weight_decay_penalty,
                                  initializer=self.initializer,
                                  isTraining=is_training,
                                  lastLayer=True)
                    self.outputs.fullFeatureList.append(result5)
                    self.outputs.generated = tf.nn.tanh(result5.cnn)
        
            
            
        if is_training and not reuse:
            self.varsTrain = [ii for ii in tf.trainable_variables() if self.scope in ii.name]
            movingMeans=[ii for ii in tf.global_variables() if self.scope in ii.name and 'moving_mean' in ii.name]
            movingVars=[ii for ii in tf.global_variables() if self.scope in ii.name and 'moving_variance' in ii.name]
            varsSave=self.varsTrain+movingMeans+movingVars
            self.saver = tf.train.Saver(max_to_keep=saveEpochs, var_list=varsSave)
            
            print(self.scope + "@" + self.config.generator.device)
            PrintNetworkVars(self.scope)
            print(print_separater)
            
            
            table = PrettyTable(['Layer', 'Act/VitOrg','Nonact/VitMap'])
            #table.add_row(['0-Encoding', str(outputBlock0.shape),str(outputFeature0.shape), 'NA'])
            # table.add_row(['1-ViT-Bridge', str(outputBlock0.shape),str(outputFeature0.shape), 'NA'])
            # table.add_row(['0-'+architectureList[0]]+result0.ProcessOutputToList())
            table.add_row(['0-Final'+architectureList[3]]+result5.ProcessOutputToList())
            table.add_row(['1-'+architectureList[3]]+result4.ProcessOutputToList())
            table.add_row(['2-'+architectureList[2]]+result3.ProcessOutputToList())
            table.add_row(['3-'+architectureList[1]]+result2.ProcessOutputToList())
            table.add_row(['4-'+architectureList[0]]+result1.ProcessOutputToList())
            # table.add_row(['6-Encoding']+final.ProcessOutputToList())
            print(table)
            print(print_separater)     
            

        return