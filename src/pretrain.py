import argparse
import os
import datetime
import yaml
import json
import gc

import numpy as np
import torch
import torch.utils
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from libs.util.model import get_vocoder, get_model
from dataset.audio_dataset import AudioDataset, AudioData
from libs.util.tool import init_random_seeds
from libs.models import FastSpeech2
from libs.models.loss_fn.fastspeech2_loss import FastSpeech2Loss, FastSpeech2LossOutput

# DDP
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(data: AudioData, device: str):
    speakers = torch.from_numpy(speakers).long().to(device)
    texts = torch.from_numpy(texts).long().to(device)
    src_lens = torch.from_numpy(src_lens).to(device)
    mels = torch.from_numpy(mels).float().to(device)
    mel_lens = torch.from_numpy(mel_lens).to(device)
    pitches = torch.from_numpy(pitches).float().to(device)
    energies = torch.from_numpy(energies).to(device)
    durations = torch.from_numpy(durations).long().to(device)

    return AudioData(
        data.ids,
        data.raw_texts,
        data.speakers.long().to(device),
        data.texts.long().to(device),
        data.src_lens.to(device),
        data.max_src_len,
        data.mels.float().to(device),
        data.mel_lens.to(device),
        data.max_mel_len,
        data.pitches.float().to(device),
        data.energies.to(device),
        data.durations.long().to(device)
    )


def train(model: FastSpeech2, data_loader: DataLoader, optimizer, grad_clip_thresh, distributed):
    losses = { key: [] for key in FastSpeech2LossOutput._fields }
    model.train()
    for batchs in data_loader:
        for batch in batchs:

            total_iters += batch[0].size(0) * args.ngpus
            batch = to_device(batch)
            
            # Forward
            output = model(*(batch[2:]))
            
            # Calculate Loss
            losses = FastSpeech2Loss(batch, output)
            
            # Backward
            total_loss = losses[0]
            total_loss.backward()
            
            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

            # Update weights
            optimizer.step_and_update_lr()
            optimizer.zero_grad()
            
            for key, val in losses._asdict().items():
                losses[key].append(val.item())
            
            torch.cuda.empty_cache()

    return { key: np.mean(val) for key, val in losses.items() }


def validate(model: FastSpeech2, data_loader: DataLoader, distributed):
    losses = { key: [] for key in FastSpeech2LossOutput._fields }
    model.eval()
    with torch.no_grad():
        for batchs in data_loader:
            for batch in batchs:
                batch = to_device(batch)
                
                # Forward
                output = model(*(batch[2:]))
                
                # Calculate Loss
                losses = FastSpeech2Loss(batch, output)
                
                for key, val in losses._asdict().items():
                    losses[key].append(val.item())
                
                torch.cuda.empty_cache()

    if distributed:
        valid_loss_logs = { key: [torch.zeros_like(val).to(args.local_rank) for _ in range(args.ngpus)] if args.local_rank == 0 else None for key, val in valid_loss_log.items() }
        for key, val in valid_loss_log.items():
            if distributed:
                torch.distributed.gather(val.to(args.local_rank), gather_list=valid_loss_logs[key], dst=0)
            else:
                valid_loss_logs[key][0] = valid_loss_log[key]
        if args.local_rank == 0:
            valid_loss_log = {
                key: torch.tensor([valid_loss_logs[key][i].cpu() for i in range(args.ngpus)]).mean()
                for key in valid_loss_log
            }

    return { key: np.mean(val) for key, val in losses.items() }


def main(args, configs):
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

    dataset = AudioDataset(
        "train.txt", preprocess_config, train_config, model_config)  # create a dataset given opt.dataset_mode and other options
    if args.local_rank == 0:
        print("Audio Size:", len(dataset))
        print("Sign  Size:", dataset.sign_size)
    valid_dataset = AudioDataset(
        "val.txt", preprocess_config, train_config, model_config)  # create a dataset given opt.dataset_mode and other options

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.ngpus, rank=args.local_rank)
        sampler_valid = torch.utils.data.distributed.DistributedSampler(
            valid_dataset, num_replicas=args.ngpus, rank=args.local_rank)
    else:
        sampler = None
        sampler_valid = None
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

    model, optimizer = get_model(args, configs, distributed, True)

    vocoder = get_vocoder(model_config, device)

    dt_now = datetime.datetime.now()
    run_name = dt_now.strftime("%m:%d:%H:%M")
    run_dir = f"../output/FastSpeech2/v2/"
    if not args.without_save_wav:
        wav_dir = run_dir + "wavs/"
        if args.local_rank == 0:
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(wav_dir, exist_ok=True)
    if args.local_rank == 0 and args.save_ckpt:
        ckpt_dir = run_dir + "ckpt/"
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

    if args.local_rank == 0 and args.use_wandb:
        wandb.init(
            project="Sign2Speech",
            group="FastSpeech2 Pretraining",
            job_type="training",
            name=run_name,
            config={
                "preprocess": preprocess_config,
                "model": model_config,
                "train": train_config
            },
        )

    total_step = train_config["step"]["total_step"]
    step_count = train_config["step"]["step_count"]
    log_step = train_config["step"]["log_step"]
    save_epochs = train_config["step"]["save_epochs"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]

    if args.local_rank == 0:
        progress = tqdm(total=total_step, desc="Training")
        progress.update(step_count)
    for epoch in range(step_count, total_step):

        train(model, loader, optimizer, grad_clip_thresh)
        validate(model, valid_loader)

        if args.local_rank == 0:
            progress.update()
        if distributed:
            torch.distributed.barrier()
        if args.local_rank == 0 and args.save_ckpt and (epoch + 1) % save_epochs == 0:
            print(f"saving the latest model {epoch=}")
            ckpt_save_path = ckpt_dir + f"{epoch}.pth"
            torch.save(
                {
                    "model": model.module.state_dict() if distributed else model.state_dict(),
                    "optimizer": optimizer._optimizer.state_dict(),
                }, ckpt_save_path)
        gc.collect()


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
        "--use_wandb", action="store_true"
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

    main(args, configs)
