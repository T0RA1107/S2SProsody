export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dataset=Sign2Speech

# Set these before running:
# CKPT_PATH=path/to/checkpoint.pth
# OUTPUT_DIR=path/to/output

python inference.py\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --ckpt_path "${CKPT_PATH}"\
 --output_dir "${OUTPUT_DIR}"

#  --without_save_wav
#  --partial_list_path dataset/test_0.txt
