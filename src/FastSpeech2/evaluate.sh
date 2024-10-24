dataset=VCTK
python evaluate.py\
 --restore_step 8000\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
 --fine_tuning --prosody
