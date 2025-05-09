import argparse
import os
import yaml
import json

import numpy as np
import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.models.semi_cycle_gan import SemiCycleGANModel
from libs.util.tool import init_random_seeds, expand
from dataset.dataset import AudioDataset, SignDataset

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

    audio_dataset = AudioDataset(
        "train.txt", preprocess_config, train_config, model_config, args)  # to get speaker info from audio train dataset
    train_dataset = SignDataset(
        preprocess_config, train_config, args.local_rank,
        phase="train", split="train"
    )
    dataset = SignDataset(
        preprocess_config, train_config, args.local_rank,
        split="test", phase="test"
    )  # create a dataset given opt.dataset_mode and other options
    speaker_info = audio_dataset.get_speaker_info()
    sign_info = train_dataset.get_sign_prosody_info()
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.ngpus, rank=args.local_rank)
    else:
        sampler = None
    if "speaker_num" not in model_config:
        model_config["speaker_num"] = audio_dataset.speaker_num
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


    model = SemiCycleGANModel(
        args, preprocess_config, model_config, train_config,
        speaker_info=speaker_info, sign_info=sign_info,
        configs_ft=configs_ft, isTrain=False, distributed=distributed)      # create a model given opt.model and other options
    model.load_networks(os.path.join(args.work_dir, args.ckpt_path))
    model.setup(train_config)               # regular setup: load and print networks; create schedulers

    border = (args.sample_num + args.ngpus - 1) // args.ngpus
    total = 0

    token_length = []
    df_wo_sign = { "duration_mean": [], "duration_std": [], "pitch_mean": [], "pitch_std": [], "energy_mean": [], "energy_std": [] }
    df_w_sign  = { "duration_mean": [], "duration_std": [], "pitch_mean": [], "pitch_std": [], "energy_mean": [], "energy_std": [] }

    if args.local_rank == 0:
        progress_bar = tqdm(range(border))
    for batchs in loader:
        for batch in batchs:
            token_length_ = batch.token_length
            token_length += token_length_.tolist()
            output_wo_sign, output_w_sign, _ = model.inference(batch)
            # without sign
            d_wo = output_wo_sign.d_rounded
            p_wo = output_wo_sign.p_predictions
            e_wo = output_wo_sign.e_predictions
            df_wo_sign["duration_mean"] += d_wo.mean(axis=1).tolist()
            df_wo_sign["pitch_mean"] += ((p_wo * d_wo).sum(axis=1) / d_wo.sum(axis=1)).tolist()
            df_wo_sign["energy_mean"] += ((e_wo * d_wo).sum(axis=1) / d_wo.sum(axis=1)).tolist()
            df_wo_sign["duration_std"] += d_wo.std(axis=1).tolist()
            df_wo_sign["pitch_std"] += [expand(p_wo[i], d_wo[i]).std() for i in range(len(d_wo))]
            df_wo_sign["energy_std"] += [expand(e_wo[i], d_wo[i]).std() for i in range(len(d_wo))]

            # with sign
            d_w = output_w_sign.d_rounded
            p_w = output_w_sign.p_predictions
            e_w = output_w_sign.e_predictions
            df_w_sign["duration_mean"] += d_w.mean(axis=1).tolist()
            df_w_sign["pitch_mean"] += ((p_w * d_w).sum(axis=1) / d_w.sum(axis=1)).tolist()
            df_w_sign["energy_mean"] += ((e_w * d_w).sum(axis=1) / d_w.sum(axis=1)).tolist()
            df_w_sign["duration_std"] += d_w.std(axis=1).tolist()
            df_w_sign["pitch_std"] += [expand(p_w[i], d_w[i]).std() for i in range(len(d_w))]
            df_w_sign["energy_std"] += [expand(e_w[i], d_w[i]).std() for i in range(len(d_w))]

            total += batch_size
            if total >= border:
                break
        if total >= border:
            break
        if args.local_rank == 0:
            progress_bar.update(group_size * batch_size)

    output_dir = os.path.join(args.work_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    token_length = np.array(token_length)
    np.save(os.path.join(output_dir, f"token_length_{args.local_rank}.npy"), token_length)
    df_wo_sign = pd.DataFrame(df_wo_sign)
    df_wo_sign.to_csv(os.path.join(output_dir, f"describe_wo_{args.local_rank}.tsv"), sep="\t", index=False)
    df_w_sign = pd.DataFrame(df_w_sign)
    df_w_sign.to_csv(os.path.join(output_dir, f"describe_w_{args.local_rank}.tsv"), sep="\t", index=False)
    if distributed:
        torch.distributed.barrier()
    if args.local_rank == 0:
        token_length = np.concatenate(
            [np.load(os.path.join(output_dir, f"token_length_{local_rank}.npy")) for local_rank in range(args.ngpus)]
        )
        np.save(os.path.join(output_dir, "token_length.npy"), token_length)
        df_wo_sign = pd.concat(
            pd.read_csv(os.path.join(output_dir, f"describe_wo_{local_rank}.tsv"), sep="\t") for local_rank in range(args.ngpus)
        )
        df_wo_sign.to_csv(os.path.join(output_dir, "describe_wo.tsv"), sep="\t", index=False)
        df_w_sign = pd.concat(
            pd.read_csv(os.path.join(output_dir, f"describe_w_{local_rank}.tsv"), sep="\t") for local_rank in range(args.ngpus)
        )
        df_w_sign.to_csv(os.path.join(output_dir, "describe_w.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--work_dir", type=str, required=True
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True
    )
    parser.add_argument(
        "--output_dir", type=str, required=True
    )
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
        "--sample_num", type=int, default=4000
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

    main(args, configs, None)
