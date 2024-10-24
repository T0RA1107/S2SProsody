# 512_pemb_bs48_ep400_encpenc_maskenc_lr3e4_ddp4_dp01_4pt_ccl10m4_e1
python train_openasl_pose_DDP_inter_VN.py \
    --ngpus 1 \
    --work_dir_prefix "/home/ubuntu/slocal/S2SProsody/src/comparison/GloFE/GloFE" \
    --work_dir "vn_model" \
    --tokenizer "notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --bs 32 \
    --prefix test-vn \
    --phase test --weights "vn_model/glofe_vn_openasl.pt" \
    --local_rank 0 \
    --label_path "/home/ubuntu/slocal/S2SProsody/data/OpenASL/data/openasl-v1.0.tsv" \
    --feat_path "/home/ubuntu/slocal/S2SProsody/data/OpenASL/data/mmpose" \
    --partial_list_path "/home/ubuntu/slocal/S2SProsody/src/comparison/GloFE/GloFE/tools/open_asl_mini.txt"
