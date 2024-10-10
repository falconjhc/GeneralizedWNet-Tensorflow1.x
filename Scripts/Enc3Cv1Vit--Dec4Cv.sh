python PipelineScripts.py --decoder EncoderCbnCbnCbnCbn  --mixer MixerMaxRes7@4  --encoder DecoderVit@2@24CbnCbnCbn  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU 
python PipelineScripts.py --decoder EncoderCbbCbbCbbCbb  --mixer MixerMaxRes7@4  --encoder DecoderVit@2@24CbbCbbCbb  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --decoder EncoderCvCvCvCv  --mixer MixerMaxRes7@4  --encoder DecoderVit@2@24CvCvCv --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU

python PipelineScripts.py --decoder EncoderCbnCbnCbnCbn  --mixer MixerMaxRes7@3  --encoder DecoderVit@2@24CbnCbnCbn  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --decoder EncoderCbbCbbCbbCbb  --mixer MixerMaxRes7@3  --encoder DecoderVit@2@24CbbCbbCbb  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --decoder EncoderCvCvCvCv  --mixer MixerMaxRes7@3  --encoder DecoderVit@2@24CvCvCv --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
