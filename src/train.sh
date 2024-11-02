dataset=Sign2Speech
python train.py\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --fine_tuning\
 --restore_step_ft 100000\
 --use_wandb
