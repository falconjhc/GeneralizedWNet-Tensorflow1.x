import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 

from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock, maxPool
from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
from Utilities.VitToolsTF import VitImplementation as vit
from Utilities.VitToolsTF import PatchMergingImplementation as patchMerger
from Utilities.VitToolsTF import PatchExpandingImplementation as patchExpander
from Utilities.utils import FindKeys


mlpRatio=4
patchSize=4

StyleFusingDict={'Max': tf.reduce_max,
                'Min': tf.reduce_min,
                'Avg': tf.reduce_mean}

class BlockFeature(object):
    def __init__(self, cnn, vit=None):
        self.cnn=cnn
        if vit is not None:
            self.vit=vit
        elif int(cnn.shape[1])>=patchSize:
            vit  = \
                tf.image.extract_patches(images=cnn, 
                                         sizes=[1, patchSize, patchSize, 1], 
                                         strides=[1, patchSize, patchSize, 1], 
                                         rates=[1, 1, 1, 1], 
                                         padding='VALID')
            patchW=int(vit.shape[1])
            patchH=int(vit.shape[2])
            self.vit = tf.reshape(vit, (vit.shape[0], patchW*patchH, patchSize*patchSize*int(cnn.shape[-1])))
        else:
            self.vit=None
    def ProcessOutputToList(self):
        if self.vit is not None:
            return [str(self.cnn.shape), str(self.vit.shape)]
        else:
            return [str(self.cnn.shape), 'None']


class EncodingBlockIO(object):
    def __init__(self, toDecoder, toNext):
        self.toDecoder=toDecoder
        self.toNext=toNext



def EncodingBasicBlock(input, blockCount, is_training, dims, weightDecay, initializer, device,  config=None, kernel=None):
    
    x = input.cnn
    _, w,h,_ = x.shape
    downsample=False
    if int(w)==int(h) and int(w)==dims['HW']*2:
        downsample=True
    
    kh=kw=3
    if kernel is not None:
        kh, kw = kernel
    
    
    identity = x
    conv1 = conv2d(x=x,
                    kh=3, kw=3, sh=1, sw=1,
                    output_filters=dims['MapC'],
                    scope="Block%d-Conv1" % blockCount,
                    parameter_update_device=device,
                    initializer=initializer,
                    weight_decay=weightDecay)
    bn1 = batch_norm(conv1, is_training, scope="Block%d-BN1" % blockCount,
                        parameter_update_device=device)
    act1 = lrelu(bn1)
    
    conv2 = conv2d(x=act1,
                    kh=kh, kw=kw, sh=1, sw=1,
                    output_filters=dims['MapC'],
                    scope="Block%d-Conv2" % blockCount,
                    parameter_update_device=device,
                    initializer=initializer,
                    weight_decay=weightDecay)
    bn2 = batch_norm(conv2, is_training, scope="Block%d-BN2" % blockCount,
                        parameter_update_device=device)
    
    if config['option'] =='Cbb':
        identity = conv2d(x=identity, output_filters=dims['MapC'], kh=1, kw=1, sh=1, sw=1, 
                            padding='SAME', parameter_update_device=device,
                            weight_decay=weightDecay,
                            initializer=initializer,
                            scope="Block%d-ResConv" % blockCount)
        identity=batch_norm(identity, is_training=is_training,
                            scope="Block%d-ResBN" % blockCount, parameter_update_device=device)
        bn2 = bn2+identity
    
    act2 = lrelu(bn2)
    toDecoder=BlockFeature(cnn=bn2)
    
    
    downsampleResult=None
    if downsample:
        downsampleResult = maxPool(act2)
        toNext=BlockFeature(cnn=downsampleResult)
    else:
        toNext=BlockFeature(cnn=act2)
    return EncodingBlockIO(toNext=toNext, toDecoder=toDecoder)

        
        
def EncodingBottleneckBlock(input, blockCount, is_training, dims, weightDecay, initializer, device,  config=None, kernel=None):
    x = input.cnn
    _, w,h,_ = x.shape
    downsample=False
    if int(w)==int(h) and int(w)==dims['HW']*2:
        downsample=True
    
    kh=kw=3
    if kernel is not None:
        kh, kw = kernel
        
    identity = x
    conv1 = conv2d(x=x,
                    kh=1, kw=1, sh=1, sw=1,
                    output_filters=dims['MapC'],
                    scope="Block%d-Conv1" % blockCount,
                    parameter_update_device=device,
                    initializer=initializer,
                    weight_decay=weightDecay)
    bn1 = batch_norm(conv1, is_training, scope="Block%d-BN1" % blockCount,
                        parameter_update_device=device)
    act1 = lrelu(bn1)
    
    conv2 = conv2d(x=act1,
                    kh=kh, kw=kw, sh=1, sw=1,
                    output_filters=dims['MapC'],
                    scope="Block%d-Conv2" % blockCount,
                    parameter_update_device=device,
                    initializer=initializer,
                    weight_decay=weightDecay)
    bn2 = batch_norm(conv2, is_training, scope="Block%d-BN2" % blockCount,
                        parameter_update_device=device)
    
    act2 = lrelu(bn2)
    
    
    conv3 = conv2d(x=act2,
                    kh=1, kw=1, sh=1, sw=1,
                    output_filters=dims['MapC'],
                    scope="Block%d-Conv3" % blockCount,
                    parameter_update_device=device,
                    initializer=initializer,
                    weight_decay=weightDecay)
    bn3 = batch_norm(conv3, is_training, scope="Block%d-BN3" % blockCount,
                    parameter_update_device=device)
    
    
    # if self.config.generator.encoder =='BottleneckEncoder':
    identity = conv2d( x=identity, output_filters=dims['MapC'], kh=1, kw=1, sh=1, sw=1, 
                        padding='SAME', parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer, 
                        scope="Block%d-ResConv" % blockCount)
    identity=batch_norm(identity, is_training=is_training,
                        scope="Block%d-ResBN" % blockCount, parameter_update_device=device)
    bn3 = bn3+identity
    act3 = lrelu(bn3)
    toDecoder=BlockFeature(cnn=bn3)
    
    
    downsampleResult=None
    if downsample:
        downsampleResult = maxPool(act3)
        toNext=BlockFeature(cnn=downsampleResult)
    else:
        toNext=BlockFeature(cnn=act3)
    return EncodingBlockIO(toNext=toNext, toDecoder=toDecoder)
        


def DecodingBasicBlock(x, dims, blockCount, device, weightDecay, initializer,  isTraining,  config=None, encLayer=None, lastLayer=False):
    
    x = x.cnn
    
    upsample=False
    _, w,h,_ = x.shape
    if int(w)==int(h) and int(w)==dims['HW']//2:
        upsample=True
        
        
    expanding=2
    if not upsample:
        expanding=1
    
    
    batchSize = int(x.shape[0])
    identity = x
    deconv1 = deconv2d(x, sh=expanding, sw=expanding, 
                       kh=3, kw=3, ######
                        output_shape=[batchSize, 
                                        int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                        scope="Block%d-DeConv1" % blockCount,
                        parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer)
    bn1 = batch_norm(deconv1, isTraining, scope="Block%d-BN1" % blockCount,
                        parameter_update_device=device)
    if not encLayer==None:
        bn1 = tf.concat([bn1, encLayer.cnn], axis=3)
    
    act1 = relu(bn1)
    deconv2 = deconv2d(act1, sh=1, sw=1,
                       kh=3, kw=3, ######
                        output_shape=[batchSize, 
                                        int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                        scope="Block%d-DeConv2" % blockCount,
                        parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer)
    bn2 = batch_norm(deconv2, isTraining, scope="Block%d-BN2" % blockCount,
                        parameter_update_device=device)
    
    if config['option']=='Cbb':
        identity = deconv2d(identity, sh=expanding, sw=expanding,
                            kh=1, kw=1, ######
                        output_shape=[batchSize, 
                                        int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                        scope="Block%d-ResDeConv" % blockCount,
                        parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer)
        identity = batch_norm(identity, isTraining, scope="Block%d-ResBN" % blockCount,
                                parameter_update_device=device)
        bn2 = bn2+identity
    act2 = relu(bn2)
    
    if not lastLayer:
        return BlockFeature(cnn=act2)
    else:
        return BlockFeature(cnn=bn2)


def DecodingBottleneckBlock(x, dims, blockCount, device, weightDecay, initializer,  isTraining,  config=None, encLayer=None, lastLayer=False):
    x = x.cnn
    
    upsample=False
    _, w,h,_ = x.shape
    if int(w)==int(h) and int(w)==dims['HW']//2:
        upsample=True
        
        
    expanding=2
    if not upsample: 
        expanding=1
    
    
    batchSize = int(x.shape[0])
    identity = x
    deconv1 = deconv2d(x, sh=expanding, sw=expanding, kh=1, kw=1,
                        output_shape=[batchSize,  
                                    int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                        scope="Block%d-DeConv1" % blockCount,
                        parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer)
    bn1 = batch_norm(deconv1, isTraining, scope="Block%d-BN1" % blockCount,
                        parameter_update_device=device)
    
    if not encLayer==None:
        bn1 = tf.concat([bn1, encLayer.cnn], axis=3)

    act1 = relu(bn1)
    
    
    deconv2 = deconv2d(act1, sh=1, sw=1, kh=3, kw=3,
                        output_shape=[batchSize, 
                                        int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                        scope="Block%d-DeConv2" % blockCount,
                        parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer)
    bn2 = batch_norm(deconv2, isTraining, scope="Block%d-BN2" % blockCount,
                        parameter_update_device=device)
    act2 = relu(bn2)
    
    
    deconv3 = deconv2d(act2, sh=1, sw=1, kh=1, kw=1,
                        output_shape=[batchSize, 
                                        int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                        scope="Block%d-DeConv3" % blockCount,
                        parameter_update_device=device,
                        weight_decay=weightDecay,
                        initializer=initializer)
    bn3 = batch_norm(deconv3, isTraining, scope="Block%d-BN3" % blockCount,
                        parameter_update_device=device)
    
    
    # if self.config.generator.decoder=='BottleneckDecoder':
    identity = deconv2d(identity, sh=expanding, sw=expanding,
                        kh=1,kw=1,################
                    output_shape=[batchSize, 
                                    int(x.shape[1]*expanding), int(x.shape[2]*expanding), dims['MapC']],
                    scope="Block%d-ResDeConv" % blockCount,
                    parameter_update_device=device,
                    weight_decay=weightDecay,
                    initializer=initializer)
    identity = batch_norm(identity, isTraining, scope="Block%d-ResBN" % blockCount,
                            parameter_update_device=device)
    bn3 = bn3+identity
    act3 = relu(bn3)
    
    if not lastLayer:
        return BlockFeature(cnn=act3)
    else:
        return BlockFeature(cnn=bn3)


def EncodingVisionTransformerBlock(input, blockCount, is_training, dims, weightDecay, initializer, device,  config=None, 
                                   downsample=True):
    
    x = input.vit
    _, dim ,_ = x.shape
    downsample=False
    
    if int(dim)==dims['VitDim']*4:
        downsample=True
    
    _, numVit, numHead = config['option'].split("@")
    numVit=int(numVit)
    numHead=int(numHead)
    batchSize = int(x.shape[0])
    
    
    vitResult = vit(x = x, scope="Block%d-ViT" % blockCount, 
                    numVit=numVit, embedDim=dims['VitC'], numHeads=numHead, ffDim=dims['VitC']*mlpRatio,
                    training=is_training)
    orgHW = dims['HW']
    if downsample: 
        orgHW=orgHW*2
    vitMap=None
    if int(vitResult.shape[1]) * int(vitResult.shape[2]) % (orgHW**2)==0:
        vitMap = tf.reshape(vitResult, (batchSize, orgHW, orgHW,-1))
    else:
        vitMap=None
    
    toDecoder=BlockFeature(cnn=vitMap, vit=vitResult)
    
    
    if downsample and not int(vitResult.shape[1])==1:
        downsampleResult = patchMerger(x =vitResult,  dim=dims['VitC'])
        downsampleMap = tf.reshape(downsampleResult, (batchSize, dims['HW'], dims['HW'],-1))
        toNext=BlockFeature(vit=downsampleResult, cnn=downsampleMap)
    elif downsample and int(vitResult.shape[1])==1:
        downsampleResult = vitResult
        downsampleMap = maxPool(vitMap)
        toNext=BlockFeature(vit=downsampleResult, cnn=downsampleMap)
        if is_training:
            print("Downsample cannot be fulfilled with vit result dimension: "+ str(vitResult.shape))
            print("Reshaped feature maps are downsampled by MaxPooling from "+str(vitMap.shape) + " to "+str(downsampleMap.shape))
    else:
        toNext=BlockFeature(cnn=vitMap, vit=vitResult)
        
    return EncodingBlockIO(toNext=toNext, toDecoder=toDecoder)
    
    

def DecodingVisionTransformerBlock(x, dims, blockCount, device, weightDecay, initializer,  isTraining,  config=None, encLayer=None, lastLayer=False):
    
    x=x.vit    
    
    _, dim ,_ = x.shape
    upsample=False
    if int(dim) == dims['VitDim']//4:
        upsample=True
    
    _, numVit, numHead = config['option'].split("@")
    if lastLayer: 
        numHead=4
    numVit=int(numVit)
    numHead=int(numHead)
    

    batchSize = int(x.shape[0])
    if len(x.shape)==4:
        if int(x.shape[1])<4:
            xVit=tf.reshape(x, (batchSize, 1, -1))
        else:
            viT1viT2 = int(x.shape[1])*int(x.shape[2])*int(x.shape[3])
            viT2 = int(x.shape[3])*patchSize*patchSize # patchSize=4
            viT1 = viT1viT2//viT2
            xVit=tf.reshape(x, (batchSize, viT1, viT2))
    elif len(x.shape)==3:
        xVit=x
    
    if upsample:
        xVit=patchExpander(x=xVit, dim=int(xVit.shape[-1]))
    if encLayer is not None:
        xVit = tf.concat([xVit, encLayer.vit], axis=-1)
    
    vitResult = vit(x = xVit, scope="Block%d-ViT" % blockCount, 
                    numVit=numVit, embedDim=dims['VitC'], numHeads=numHead, ffDim=dims['VitC']*mlpRatio,
                    training=isTraining)
    
    resultC = int(vitResult.shape[-1])//(patchSize*patchSize)
    resultHW = int(np.sqrt(int(vitResult.shape[1])*int(vitResult.shape[2])//resultC))
    mapResult=tf.reshape(vitResult, (batchSize, resultHW, resultHW,resultC))
    
    return BlockFeature(cnn=mapResult, vit=vitResult)
    



def FusingStyleFeatures(repeatNum, fusingList, fusionMethod, needAct, architecture, is_training, scope, weightDecay, initializer, device, outputMark='NA'):
    
    BlockEncDict={'Cv': EncodingBottleneckBlock, 
                  'Cbb': EncodingBottleneckBlock, 
                  'Cbn': EncodingBottleneckBlock, 
                  'Vit': EncodingVisionTransformerBlock}

           
    for ii in range(repeatNum):
        thisCNN = tf.expand_dims(fusingList[ii][0].cnn, axis=0)
        thisVit = tf.expand_dims(fusingList[ii][0].vit, axis=0)
        if ii==0:
            fusedCNN=thisCNN
            fusedVit=thisVit
        else:
            fusedCNN=tf.concat([fusedCNN, thisCNN], axis=0)
            fusedVit=tf.concat([fusedVit, thisVit], axis=0)
            
    if fusionMethod=='Max' or fusionMethod=='Min' or fusionMethod=='Avg':
        fusedCNN = StyleFusingDict[fusionMethod](fusedCNN,axis=0)
        fusedVit = StyleFusingDict[fusionMethod](fusedVit,axis=0)
        fusedFinalStyle=BlockFeature(cnn=fusedCNN, vit=fusedVit)
        return fusedFinalStyle

    elif fusionMethod=='Adaptive':
        fusedCNN = tf.reshape(fusedCNN, (int(fusedCNN.shape[1]),
                                        int(fusedCNN.shape[2]),int(fusedCNN.shape[3]),
                                        int(fusedCNN.shape[0])*int(fusedCNN.shape[4])))    
        cnnC1, cnnC2, cnnC3 = int(fusedCNN.shape[-1])//3*2, \
            int(fusedCNN.shape[-1])//3, \
                int(fusedCNN.shape[-1])//repeatNum      
                
                
        fusedVit = tf.reshape(fusedVit, (int(fusedVit.shape[1]),int(fusedVit.shape[2]),
                                        int(fusedVit.shape[0])*int(fusedVit.shape[3])))  
        _, numVit, numHead =  architecture.split("@") 
        vitC1, vitC2, vitC3 = int(fusedVit.shape[-1])//3*2, \
                int(fusedVit.shape[-1])//3, \
                    int(fusedVit.shape[-1])//repeatNum



        with tf.variable_scope(scope):
            if needAct:
                fusedCNN=lrelu(fusedCNN)
            fusedFinalStyle=BlockFeature(cnn=fusedCNN, vit=fusedVit)
            thisBlock = FindKeys(BlockEncDict, architecture)[0]
            cnnHW=int(fusedFinalStyle.cnn.shape[1])
            vitDim = int(fusedFinalStyle.vit.shape[1])
            
            fusedFinalStyle=thisBlock(input=fusedFinalStyle,  
                                            blockCount=1, 
                                            is_training=is_training,  
                                            dims={'HW': cnnHW, 'MapC': cnnC1,
                                                'VitC': vitC1,'VitDim': vitDim}, 
                                            config={'option': architecture,
                                                    'numViT': numVit,
                                                    'numHead': numHead},
                                            weightDecay=weightDecay, 
                                            initializer=initializer,  
                                            device=device)
            fusedFinalStyle=thisBlock(input=fusedFinalStyle.toNext,  
                                            blockCount=2, 
                                            is_training=is_training,  
                                            dims={'HW': cnnHW, 'MapC': cnnC2,
                                                'VitC': vitC2, 'VitDim': vitDim}, 
                                            config={'option': architecture,
                                                    'numViT': numVit,
                                                    'numHead': numHead},
                                            weightDecay=weightDecay, 
                                            initializer=initializer,  
                                            device=device)
            fusedFinalStyle=thisBlock(input=fusedFinalStyle.toNext,  
                                            blockCount=3, 
                                            is_training=is_training,  
                                            dims={'HW': cnnHW,  'MapC': cnnC3,
                                                'VitC': vitC3, 'VitDim': vitDim}, 
                                            config={'option': architecture,
                                                    'numViT': numVit,
                                                    'numHead': numHead},
                                            weightDecay=weightDecay, 
                                            initializer=initializer,  
                                            device=device)
        if outputMark == 'toDecoder':
            fusedFinalStyle=fusedFinalStyle.toDecoder
        elif outputMark == 'toNext':
            fusedFinalStyle=fusedFinalStyle.toNext
        return fusedFinalStyle