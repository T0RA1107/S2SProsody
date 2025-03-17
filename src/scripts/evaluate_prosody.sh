export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dataset=Sign2Speech
python evaluate_prosody.py\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --work_dir "/home/aolab/Desktop/S2SProsody/src/output/newReg"\
 --ckpt_path "ckpt/17.pth"\
 --output_dir "eval/"\
 --sample_num 800
