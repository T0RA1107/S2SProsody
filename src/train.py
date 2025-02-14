import argparse
import os
import datetime
import random
import yaml
import json

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from FastSpeech2.evaluate import evaluate
from FastSpeech2.utils.model import get_vocoder

from models.semi_cycle_gan import SemiCycleGANModel
from dataset import UnpairedAudioSignDataset, SignDataset
from util.tool import init_random_seeds, save_inference, save_metadata, save_validation_loss

# DDP
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs, configs_ft):
    preprocess_config, model_config, train_config = configs

    # [Steup]
    if args.ngpus > 1:
        # init DDP
        distributed = True
        dist.init_process_group(backend="nccl")
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
    else:
        # on process, treat as the master process
        args.local_rank = 0
        distributed = False

    # Ensure each process has the same initialization
    init_random_seeds(args.seed, args.local_rank)

    dataset = UnpairedAudioSignDataset(
        "train.txt", preprocess_config, train_config, model_config, args)  # create a dataset given opt.dataset_mode and other options
    valid_dataset = SignDataset(
        args, preprocess_config, train_config,
        phase="train", split="val"
    )
    inference_dataset = SignDataset(
        args, preprocess_config, train_config,
        phase="test", split="val", partial_list_path="./valid_list.txt"
    )
    speaker_info = dataset.audio_dataset.get_speaker_info()
    sign_info = dataset.sign_dataset.get_sign_prosody_info()
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.ngpus, rank=args.local_rank)
        sampler_valid = torch.utils.data.distributed.DistributedSampler(
            valid_dataset, num_replicas=args.ngpus, rank=args.local_rank)
        sampler_inference = torch.utils.data.distributed.DistributedSampler(
            inference_dataset, num_replicas=args.ngpus, rank=args.local_rank)
    else:
        sampler = None
        sampler_valid = None
        sampler_inference = None
    if "speaker_num" not in model_config:
        model_config["speaker_num"] = dataset.speaker_num
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size * group_size,
        shuffle=(sampler_valid is None),
        sampler=sampler_valid,
        collate_fn=valid_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size * group_size,
        shuffle=(sampler_inference is None),
        sampler=sampler_inference,
        collate_fn=inference_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )

    model = SemiCycleGANModel(
        args, preprocess_config, model_config, train_config,
        speaker_info=speaker_info, sign_info=sign_info,
        configs_ft=configs_ft, distributed=distributed)      # create a model given opt.model and other options
    model.setup(train_config)               # regular setup: load and print networks; create schedulers

    vocoder = get_vocoder(model_config, device)

    dt_now = datetime.datetime.now()
    run_name = dt_now.strftime("%m:%d:%H:%M")
    run_dir = f"./output/{run_name}/"
    if not args.without_save_wav:
        wav_dir = run_dir + "wavs/"
        if args.local_rank == 0:
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)
    if args.local_rank == 0 and args.save_ckpt:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(run_dir + "ckpt", exist_ok=True)

    if args.local_rank == 0 and args.use_wandb:
        wandb.init(
            project="Sign2Speech",
            group="Reg on Prosody Dist",
            job_type="training",
            name=run_name,
            config={
                "preprocess": preprocess_config,
                "model": model_config,
                "train": train_config
            },
        )

    total_iters = 0
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    n_epochs_decay = train_config["step"]["n_epochs_decay"]
    step_count = train_config["step"]["step_count"]
    log_step = train_config["step"]["log_step"]
    save_epochs = train_config["step"]["save_epochs"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    assert synth_step % log_step == 0
    assert val_step % log_step == 0

    if args.local_rank == 0:
        progress = tqdm(total=len(range(step_count, total_step + n_epochs_decay)), desc="Training")
        nxt_log_step = log_step
    for epoch in range(step_count, total_step + n_epochs_decay):    # outer loop for different epochs.
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        if distributed:
            sampler.set_epoch(epoch)
        for batchs in loader:  # inner loop within one epoch
            for batch in batchs:

                total_iters += 1
                model.set_input(batch)         # unpack data from dataset and apply preprocessing
                loss_log = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                if args.local_rank == 0 and total_iters >= nxt_log_step:    # print training losses and save logging information to the disk
                    log = { "epoch": epoch }
                    lr_dict = model.get_learning_rate()
                    log.update(lr_dict)
                    log.update(loss_log)
                    if args.use_wandb:
                        wandb.log(log)
                    nxt_log_step += log_step

        if args.local_rank == 0:
            progress.update()
        if args.use_wandb:
            torch.distributed.barrier()
            valid_loss_log = save_validation_loss(model, valid_loader)
            valid_loss_logs = { key: [torch.zeros_like(val).to(args.local_rank) for _ in range(args.ngpus)] if args.local_rank == 0 else None for key, val in valid_loss_log.items() }
            for key, val in valid_loss_log.items():
                torch.distributed.gather(val.to(args.local_rank), gather_list=valid_loss_logs[key], dst=0)
            if args.local_rank == 0:
                valid_loss_log = {
                    key: torch.tensor([valid_loss_logs[key][i].cpu() for i in range(args.ngpus)]).mean()
                    for key in valid_loss_log
                }
                wandb.log(valid_loss_log)
            del valid_loss_log, valid_loss_logs
            torch.cuda.empty_cache()
        if not args.without_save_wav:
            torch.distributed.barrier()
            save_inference(model, vocoder, inference_loader, args.local_rank, sampling_rate, stats, wav_dir, epoch, model_config, preprocess_config)
            torch.distributed.barrier()
            if args.local_rank == 0:
                save_metadata(args.ngpus, wav_dir, epoch)
        if args.local_rank == 0 and args.save_ckpt and (epoch + 1) % save_epochs == 0:
            print(f"saving the latest model {epoch=}")
            ckpt_save_path = run_dir + f"ckpt/{epoch}.pth"
            model.save_networks(ckpt_save_path)
        if args.ngpus > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--isTrain", action="store_true"
    )
    parser.add_argument(
        "--use_wandb", action="store_true"
    )
    parser.add_argument(
        "--fine_tuning", action="store_true"
    )
    parser.add_argument(
        "--restore_step_ft", type=int
    )
    parser.add_argument(
        "--save_ckpt", action="store_true"
    )
    parser.add_argument(
        "--without_save_wav", action="store_true"
    )
    # DDP related
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngpus", type=int, default=1,
                        help="number of gpus used, equivilent to world_size(local)")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="pass rank info throughout the script, DO NOT input through command line")
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    configs_ft = None
    if args.fine_tuning:
        ft_config_path = train_config["fine-tuning"]
        preprocess_config_ft = yaml.load(open(
            os.path.join(ft_config_path, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
        model_config_ft = yaml.load(open(
            os.path.join(ft_config_path, "model.yaml"), "r"), Loader=yaml.FullLoader)
        train_config_ft = yaml.load(open(
            os.path.join(ft_config_path, "train.yaml"), "r"), Loader=yaml.FullLoader)
        configs_ft = (preprocess_config_ft, model_config_ft, train_config_ft)

    main(args, configs, configs_ft)
