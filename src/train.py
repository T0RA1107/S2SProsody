import argparse
import datetime
import yaml
import json
import gc
from pathlib import Path
from logging import getLogger, basicConfig, DEBUG, INFO

import torch
import torch.utils
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import wandb
from tqdm import tqdm

from libs.models import SemiCycleGANModel, SemiCycleGANModelWrapper
from datasets import UnpairedAudioSignDataset, SignDataset
from libs.util.model import get_vocoder
from libs.util.tool import init_random_seeds
from libs.util.save_data import save_inference, save_metadata, save_validation_loss
from libs.util.logger import TrainLogger

# DDP
import torch.distributed as dist

logger = getLogger(__name__)


def main(args, configs, configs_ft):
    # torch.autograd.set_detect_anomaly(True)
    preprocess_config, model_config, train_config = configs

    dt_now = datetime.datetime.now()
    run_name = dt_now.strftime("%m:%d:%H:%M")
    result_dir = Path(train_config["path"]["result_path"], run_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=result_dir / "train.log",
    )
    train_logger = TrainLogger(result_dir)

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
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # Ensure each process has the same initialization
    init_random_seeds(args.seed, args.local_rank)

    dataset = UnpairedAudioSignDataset(
        "train.txt", preprocess_config, train_config, model_config, args.local_rank)  # create a dataset given opt.dataset_mode and other options
    if args.local_rank == 0:
        logger.info(f"Audio Size: {dataset.audio_size}")
        logger.info(f"Sign  Size: {dataset.sign_size}")
    valid_dataset = SignDataset(
        preprocess_config, train_config, args.local_rank,
        phase="train", split="val"
    )
    inference_dataset = SignDataset(
        preprocess_config, train_config, args.local_rank,
        phase="test", split="val", partial_list_path="./datasets/valid_list.txt"
    )
    speaker_info = dataset.audio_dataset.get_speaker_info()
    sign_info = dataset.sign_dataset.get_sign_prosody_info()
    with Path(preprocess_config["path"]["preprocessed_path"], "stats.json").open() as f:
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
    num_workers = 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size * group_size,
        shuffle=(sampler_valid is None),
        sampler=sampler_valid,
        collate_fn=valid_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size * group_size,
        shuffle=(sampler_inference is None),
        sampler=sampler_inference,
        collate_fn=inference_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    model = SemiCycleGANModel(
        args, preprocess_config, model_config, train_config,
        speaker_info=speaker_info, sign_info=sign_info,
        configs_ft=configs_ft, distributed=distributed)      # create a model given opt.model and other options

    model.setup(train_config, len(loader) * group_size)               # regular setup: load and print networks; create schedulers
    model_wrapper = SemiCycleGANModelWrapper(model)
    vocoder = get_vocoder(model_config, device)
    if distributed:
        vocoder = DistributedDataParallel(
                vocoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )

    output_dir = Path(train_config["path"]["output_path"], run_name)
    if not args.without_save_wav:
        wav_dir = output_dir / "wavs"
    if args.save_ckpt:
        ckpt_dir = output_dir / "ckpt"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

    if args.local_rank == 0 and args.use_wandb:
        wandb.init(
            project="Sign2Speech",
            group=args.group,
            job_type="training",
            name=run_name,
            config={
                "preprocess": preprocess_config,
                "model": model_config,
                "train": train_config
            },
        )

    total_iters = 0
    total_step = train_config["step"]["total_step"]
    step_count = train_config["step"]["step_count"]
    log_step = train_config["step"]["log_step"]
    save_epochs = train_config["step"]["save_epochs"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    if args.local_rank == 0:
        progress = tqdm(total=total_step, desc="Training")
        progress.update(step_count)
        nxt_log_step = log_step
    for epoch in range(step_count, total_step):    # outer loop for different epochs.
        if distributed:
            sampler.set_epoch(epoch)
        for batchs in loader:  # inner loop within one epoch
            for batch in batchs:

                total_iters += batch_size * args.ngpus
                model.set_input(batch)         # unpack data from dataset and apply preprocessing
                loss_log = model.optimize_parameters(epoch < train_config["optimizer"]["discriminator_epoch"])   # calculate loss functions, get gradients, update network weights

                if args.local_rank == 0 and total_iters >= nxt_log_step:    # print training losses and save logging information to the disk
                    log = { "epoch": epoch }
                    lr_dict = model.get_learning_rate()
                    log.update(lr_dict)
                    log.update(loss_log)
                    train_logger.update(log)
                    if args.use_wandb:
                        wandb.log(log)
                    nxt_log_step += log_step
                model.update_learning_rate()

        if args.local_rank == 0:
            progress.update()
        # validation
        if distributed:
            torch.distributed.barrier()
        valid_loss_log = save_validation_loss(model_wrapper, valid_loader)
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
            if args.use_wandb:
                wandb.log(valid_loss_log)
            logger.info(valid_loss_log)
        del valid_loss_log, valid_loss_logs
        torch.cuda.empty_cache()
        # save wav file
        if not args.without_save_wav:
            if distributed:
                torch.distributed.barrier()
            save_inference(model_wrapper, vocoder, inference_loader, args.local_rank, sampling_rate, stats, wav_dir, epoch, model_config, preprocess_config, inference_dataset.partial_vid2gender)
            if distributed:
                torch.distributed.barrier()
            if args.local_rank == 0:
                save_metadata(args.ngpus, wav_dir, epoch)
            torch.cuda.empty_cache()
        # save model weight
        if args.local_rank == 0 and args.save_ckpt and (epoch + 1) % save_epochs == 0:
            logger.info(f"saving the latest model {epoch=}")
            model.save_networks((ckpt_dir / f"{epoch}.pth"))
        if distributed:
            torch.distributed.barrier()
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
        "-d", "--debug", action="store_true"
    )
    # W & B related
    parser.add_argument(
        "--use_wandb", action="store_true"
    )
    parser.add_argument(
        "--group", type=str
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
        Path(args.preprocess_config).open(), Loader=yaml.FullLoader
    )
    model_config = yaml.load(Path(args.model_config).open(), Loader=yaml.FullLoader)
    train_config = yaml.load(Path(args.train_config).open(), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    configs_ft = None
    if args.fine_tuning:
        ft_config_path = train_config["fine-tuning"]
        preprocess_config_ft = yaml.load(Path(ft_config_path, "preprocess.yaml").open(), Loader=yaml.FullLoader)
        model_config_ft = yaml.load(Path(ft_config_path, "model.yaml").open(), Loader=yaml.FullLoader)
        train_config_ft = yaml.load(Path(ft_config_path, "train.yaml").open(), Loader=yaml.FullLoader)
        configs_ft = (preprocess_config_ft, model_config_ft, train_config_ft)

    main(args, configs, configs_ft)
