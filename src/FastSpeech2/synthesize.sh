poetry shell
dataset=VCTK
python synthesize.py\
 --text "And it's difficult to argue."\
 --speaker_id 8 --restore_step 10000\
 --mode single\
 -p config/${dataset}/preprocess.yaml\
 -m config/${dataset}/model.yaml\
 -t config/${dataset}/train.yaml\
