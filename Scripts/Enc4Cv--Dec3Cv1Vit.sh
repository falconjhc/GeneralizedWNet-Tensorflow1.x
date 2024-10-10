
python PipelineScripts.py --encoder EncoderCbnCbnCbnCbn  --mixer MixerMaxRes7@4  --decoder DecoderVit@2@24CbnCbnCbn  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCbbCbbCbbCbb  --mixer MixerMaxRes7@4  --decoder DecoderVit@2@24CbbCbbCbb  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCvCvCvCv  --mixer MixerMaxRes7@4  --decoder DecoderVit@2@24CvCvCv --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU


python PipelineScripts.py --encoder EncoderCbnCbnCbnCbn  --mixer MixerMaxRes7@3  --decoder DecoderVit@2@24CbnCbnCbn  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCbbCbbCbbCbb  --mixer MixerMaxRes7@3  --decoder DecoderVit@2@24CbbCbbCbb  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCvCvCvCv  --mixer MixerMaxRes7@3  --decoder DecoderVit@2@24CvCvCv --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
