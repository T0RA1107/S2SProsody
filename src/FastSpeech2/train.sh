poetry shell
dataset=VCTK
python train.py\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --fine_tuning\
 --restore_step_ft 800000\
