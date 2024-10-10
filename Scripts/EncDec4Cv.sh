python PipelineScripts.py --encoder EncoderCbnCbnCbnCbn  --mixer MixerMaxRes7@4  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU --skipTest True
python PipelineScripts.py --encoder EncoderCbbCbbCbbCbb  --mixer MixerMaxRes7@4  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCvCvCvCv  --mixer MixerMaxRes7@4  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU

python PipelineScripts.py --encoder EncoderCbnCbnCbnCbn  --mixer MixerMaxRes7@3  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU --skipTest True
python PipelineScripts.py --encoder EncoderCbbCbbCbbCbb  --mixer MixerMaxRes7@3  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCvCvCvCv  --mixer MixerMaxRes7@3  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU