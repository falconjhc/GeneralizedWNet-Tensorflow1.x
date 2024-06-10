from tensorboard.backend.event_processing import event_accumulator        # 导入tensorboard的事件解析器
import os 
import pandas as pd
import re
# import sets


# WritingFileTrain='./TrainRecordsTestVitEncoder-20240517-Train.xlsx'
# WritingFileValidate='./TrainRecordsTestVitEncoder-20240517-Validate.xlsx'

WritingFile='./TrainRecordsTestVitEncoder-20240517.xlsx'
DelNames=['Exp20240517-ContentPF64-StylePF50-',
          'Exp20240518-ContentPF64-StylePF50-',
          'Exp20240519-ContentPF64-StylePF50-',
          '-AvgMaxConvResidualMixer-BasicBlockDecoder',
          '_Encoder']

logDir = "/data-shared/server02/data1/haochuan/Character/TrainRecordsTestVitEncoder-20240517/Logs/"


checkingEvals=[
    'LossDeepPerceptual-ContentMSE', 
    'LossDeepPerceptual-ContentVN',
    'LossDeepPerceptual-StyleMSE',
    'LossDeepPerceptual-StyleVN',
    'LossReconstruction'
]

replaceModelNames=dict()
replaceModelNames.update({'CV_CV_CV_CV_CV': "5CV",
                          'CV_CV_CV_CV_ViT': "4CV-1ViT",
                          'CV_CV_CV_ViT_ViT': "3CV-2ViT",
                          'CV_CV_ViT_ViT_ViT': "2CV-3ViT",
                          'CV_ViT_ViT_ViT_ViT': "1CV-4ViT",

                          "CBN_CBN_CBN_CBN_CBN": "5CBN",
                          'CBN_CBN_CBN_CBN_ViT': "4CBN-1ViT",
                          'CBN_CBN_CBN_ViT_ViT': "3CBN-2ViT",
                          'CBN_CBN_ViT_ViT_ViT': "2CBN-3ViT",
                          'CBN_ViT_ViT_ViT_ViT': "1CBN-4ViT",

                          "CBB_CBB_CBB_CBB_CBB": "5CBB",
                          'CBB_CBB_CBB_CBB_ViT': "4CBB-1ViT",
                          'CBB_CBB_CBB_ViT_ViT': "3CBB-2ViT",
                          'CBB_CBB_ViT_ViT_ViT': "2CBB-3ViT",
                          'CBB_ViT_ViT_ViT_ViT': "1CBB-4ViT",
})


def replace_all_digits(string):
    digits = re.findall(r'\d', string)
    for digit in digits:
        string = string.replace(digit, '')
    return string


if os.path.exists(WritingFile):
    os.remove(WritingFile)
# if os.path.exists(WritingFileValidate):
#     os.remove(WritingFileValidate)
    
eventCount=0


# Find all events and all logs
evalCounter=0
full=dict()
for root, dirs, files in os.walk(logDir):
    for name in files:
        if 'tfevents' in name:
            eventCount+=1
            tfEventFile=os.path.join(root, name)
            expName=tfEventFile.split('/')[-2:-1][0]
            for name in DelNames:
                expName=expName.replace(name,'')
            ea=event_accumulator.EventAccumulator(tfEventFile)     # 初始化EventAccumulator对象
            ea.Reload()    # 这一步是必须的，将事件的内容都导进去
            logs = ea.scalars.Keys()
            for log in logs:
                logPack=ea.scalars.Items(log)
                if log not in full:
                    for eval in checkingEvals:
                        if eval in log:
                            evalCounter+=1
                            full.update({log:dict()})
                            # continue
                            print("%d-%d-Checking Evaluations: %s" % (evalCounter, eventCount, log))
                    # evalCounter+=1
                    # full.update({log:dict()})
                    # print("%d-%d-Checking Evaluations: %s" % (evalCounter, eventCount, log))
    if eventCount>5:
        break    
                
            
            

eventCount=0
for root, dirs, files in os.walk(logDir):
    for name in files:
        if 'tfevents' in name:
            eventCount+=1
            tfEventFile=os.path.join(root, name)
            expName=tfEventFile.split('/')[-2:-1][0]
            # if 'Exp20240519-ContentPF64-StylePF50-TransWNet-AvgMaxConvResidualMixer-BasicBlockDecoder-CBB_CBB_CBB_CBB_CBB_Encoder' == expName:
            #     continue
            for name in DelNames:
                expName=expName.replace(name,'')
            ea=event_accumulator.EventAccumulator(tfEventFile)     # 初始化EventAccumulator对象
            ea.Reload()    # 这一步是必须的，将事件的内容都导进去
            logs = ea.scalars.Keys()
            # event = Event(tfEventFile)
            print("Collecting %03d evaluation of %s" % (eventCount, expName))
            steps=list()
            for log in logs:
                if not log in full.keys():
                    continue
                logPack=ea.scalars.Items(log)
                # log =log.replace('/','-')
                values=list()
                if len(steps)==0:
                    for ii in logPack:
                        steps.append(ii.step)
                if not 'steps' in full[log].keys():
                    full[log].update({'Steps': steps})
                for ii in logPack:
                    values.append(ii.value)
                expNameNew = replace_all_digits(expName)
                expNameNew=expNameNew.replace("@", "")
                for key, value in replaceModelNames.items():
                    expNameNew=expNameNew.replace(key,value)
                full[log].update({expNameNew: values})
                
    # if eventCount>3:
    #     break   
                
counter=0
for evaluate in full.keys():
    # if 'Train' in evaluate:
    #     WritingFile=WritingFileTrain
    # elif 'Val' in evaluate:
    #     WritingFile=WritingFileValidate
    if counter==0:
        writer= pd.ExcelWriter(WritingFile, engine='openpyxl')
    else:
        writer= pd.ExcelWriter(WritingFile, engine='openpyxl', mode='a')
    data=pd.DataFrame(full[evaluate])
    
    evaluate=evaluate.replace('Train','Trn')
    evaluate=evaluate.replace('Validate','Val')
    evaluate=evaluate.replace('/','-')
    evaluate=evaluate.replace('--','-')
    evaluate=evaluate.replace('-','')
    evaluate=evaluate.replace('Loss','')
    evaluate=evaluate.replace('DeepPerceptual','DP')
    evaluate=evaluate.replace('Reconstruction','')
    evaluate=evaluate.replace('Content','CNT')
    evaluate=evaluate.replace('Style','STY')
    evaluate=evaluate.replace('net','')
    evaluate=evaluate.replace('vgg','Vgg')
    evaluate=evaluate.replace('resnet','Res')
    # evaluate=replace_all_digits(evaluate)
    data.to_excel(writer, sheet_name=evaluate.replace('/','-'), index=False)
    writer.save()#保存
    
    counter+=1
    print("%d-%s evaluation saved" % (counter, evaluate))

print("Complete! ")

