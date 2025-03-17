export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dataset=Sign2Speech
python inference.py\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --ckpt_path /home/aolab/Desktop/S2SProsody/src/output/pitch/ckpt/39.pth

#  --without_save_wav
