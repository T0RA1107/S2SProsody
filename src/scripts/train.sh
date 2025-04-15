# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dataset=Sign2Speech
GPUS=2
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$GPUS train.py\
 --ngpus $GPUS\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --fine_tuning\
 --restore_step_ft 1000000\
 --use_wandb\
 --group "MoE"\
 --save_ckpt
#  --without_save_wav
