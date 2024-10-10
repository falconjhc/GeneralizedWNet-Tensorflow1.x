
import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'seed':1,
        'debugMode': 0,
        'expID':'20240724-ContentPF64-StylePF50',# experiment name prefix
        'expDir': '/data-shared/server02/data1/haochuan/Character/TrainRecords-202407/',
        


        # devices
        'generatorDevice':'/device:GPU:0',
        'discrminatorDevice':'/device:GPU:0',
        'featureExtractorDevice':'/device:GPU:0',

        
        # data 
        'content_data_dir':# standard data location
        [os.path.join(dataPathRoot, 'CASIA_Dataset/PrintedData_64Fonts/Simplified/GB2312_L1/')],

        'style_train_data_dir': # training data location
        [os.path.join(dataPathRoot, 'CASIA_Dataset/PrintedData/GB2312_L1/')],

        'style_validation_data_dir':# validation data location
        [os.path.join(dataPathRoot, 'CASIA_Dataset/PrintedData/GB2312_L1/')],

        'file_list_txt_content': # file list of the standard data
        ['../FileList/PrintedData/Char_0_3754_64PrintedFonts_GB2312L1_Simplified.txt'],

        'file_list_txt_style_train': # file list of the training data
        ['../TrainTestFileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1_Train.txt'],

        'file_list_txt_style_validation': # file list of the validation data
        ['../TrainTestFileList/PrintedData/Char_0_3754_Font_50_79_GB2312L1_Test.txt'],

        'FullLabel0Vec': 'CASIA_Dataset/LabelVecs/PF80-Label0.txt',
        'FullLabel1Vec': 'CASIA_Dataset/LabelVecs/PF80-Label1.txt',
        
        
        # training configurations
        'trainAugmentContentTranslation':1,
        'trainAugmentContentRotation':1,
        'trainAugmentContentFlip':1,
        'trainAugmentStyleTranslation':1,
        'trainAugmentStyleRotation':1,
        'trainAugmentStyleFlip':1,
        'trainSplitContentStyleAugmentation': 1,

        'testAugmentContentTranslation':0,
        'testAugmentContentRotation':0,
        'testAugmentContentFlip':0,
        'testAugmentStyleTranslation':0,
        'testAugmentStyleRotation':0,
        'testAugmentStyleFlip':0,
        'testSplitContentStyleAugmentation':0,
        
        'inputStyleNum':5, 

        # generator && discriminator
        'generator': 'TransWNet-AvgMaxConvResidualMixer-BasicBlockDecoder',
        'discriminator':'NA',

        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimization_method':'adam',
        'initTrainEpochs':0,
        'final_learning_rate_pctg': 0.30,

        # feature extractor parametrers
        'true_fake_target_extractor_dir': [],
        
        'content_prototype_extractor_dir':
        [
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_VGG16-FeatureExtractor_Content_PF64_vgg16net/variables/@/device:GPU:1',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_VGG19-FeatureExtractor_Content_PF64_vgg19net/variables/@/device:GPU:1',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet18-FeatureExtractor_Content_PF64_resnet18/variables/@/device:GPU:1',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet34-FeatureExtractor_Content_PF64_resnet34/variables/@/device:GPU:1',
        #'/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet50-FeatureExtractor_Content_PF64_resnet50/variables/@/device:GPU:1',
        #'/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF64/Models/checkpoint/Exp20240308_ResNet101-FeatureExtractor_Content_PF64_resnet101/variables/@/device:GPU:1'
        ],
        
        'style_reference_extractor_dir':
       [
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF80/Models/checkpoint/Exp20240308_VGG16-FeatureExtractor_Content_PF80_vgg16net/variables/@/device:GPU:1',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF80/Models/checkpoint/Exp20240308_VGG19-FeatureExtractor_Content_PF80_vgg19net/variables//@/device:GPU:1',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF80/Models/checkpoint/Exp20240308_ResNet18-FeatureExtractor_Content_PF80_resnet18/variables/@/device:GPU:1',
        '/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF80/Models/checkpoint/Exp20240308_ResNet34-FeatureExtractor_Content_PF80_resnet34/variables/@/device:GPU:1',
        #'/data-shared/server02/data1/haochuan/Character/2024-FeatureExtractorDemonstration/PF80/Models/checkpoint/Exp20240308_ResNet50-FeatureExtractor_Content_PF80_resnet50/variables/@/device:GPU:0'
        
        ]

}


penalties = {
        'generator_weight_decay_penalty': 0.0001,
        'discriminator_weight_decay_penalty':0.0003,
        'Pixel_Reconstruction_Penalty':1,
        'Lconst_content_Penalty':0.2,
        'Lconst_style_Penalty':0.2,
        'Discriminative_Penalty': 0,
        'Discriminator_Categorical_Penalty': 0,
        'Generator_Categorical_Penalty': 0.,
        'Discriminator_Gradient_Penalty': 0,
        'Batch_StyleFeature_Discrimination_Penalty':0,
        'FeatureExtractorPenalty_ContentPrototype': [0.5, 0.3, 0.3,0.2],
        'FeatureExtractorPenalty_StyleReference':[1, 0.5, 0.3,0.2]
}