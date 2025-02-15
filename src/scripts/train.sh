# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dataset=Sign2Speech
GPUS=1
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=$GPUS train.py\
 --ngpus $GPUS\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --fine_tuning\
 --restore_step_ft 1000000\
 --without_save_wav
#  --save_ckpt\
#  --use_wandb
#  --without_save_wav
