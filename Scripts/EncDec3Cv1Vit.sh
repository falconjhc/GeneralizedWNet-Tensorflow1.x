python PipelineScripts.py --encoder EncoderCbnCbnCbnVit@2@24  --mixer MixerMaxRes7@4  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCbbCbbCbbVit@2@24  --mixer MixerMaxRes7@4  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCvCvCvVit@2@24  --mixer MixerMaxRes7@4  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU

python PipelineScripts.py --encoder EncoderCbnCbnCbnVit@2@24  --mixer MixerMaxRes7@3  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCbbCbbCbbVit@2@24  --mixer MixerMaxRes7@3  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
python PipelineScripts.py --encoder EncoderCvCvCvVit@2@24  --mixer MixerMaxRes7@3  --batchSize 12 --initLr 0.001 --epochs 21 --resumeTrain 0 --config PF64-HW50-1GPU
