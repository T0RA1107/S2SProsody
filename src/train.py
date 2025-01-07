import argparse
import os
import datetime
import random

from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import soundfile as sf
import torch
import torch.utils
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from FastSpeech2.evaluate import evaluate
from FastSpeech2.utils.model import get_vocoder, vocoder_infer

from models.semi_cycle_gan import SemiCycleGANModel
from dataset import UnpairedAudioSignDataset

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_random_seeds(random_seed=0, rank=0):
    # eliminate isomophisim across ranks
    the_seed = random_seed + rank
    torch.manual_seed(the_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(the_seed)
    np.random.seed(the_seed)
    random.seed(the_seed)
    # These two slows down training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def main(args, configs, configs_ft):
    preprocess_config, model_config, train_config = configs

    # [Steup]
    if args.ngpus > 1:
        # init DDP
        distributed = True
        dist.init_process_group(backend='nccl')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
    else:
        # on process, treat as the master process
        args.local_rank = 0
        distributed = False

    # Ensure each process has the same initialization
    init_random_seeds(args.seed, args.local_rank)

    dataset = UnpairedAudioSignDataset(
        "train.txt", preprocess_config, train_config, args)  # create a dataset given opt.dataset_mode and other options
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.ngpus, rank=args.local_rank)
    else:
        sampler = None
    if "speaker_num" not in model_config:
        model_config["speaker_num"] = dataset.speaker_num
    dataset_size = len(dataset)    # get the number of images in the dataset.
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
    if args.local_rank == 0:
        print("Batch size:", batch_size)
        print('The number of training images = %d' % dataset_size)

    model = SemiCycleGANModel(args, preprocess_config, model_config, train_config, configs_ft, distributed=distributed)      # create a model given opt.model and other options
    model.setup(train_config)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    vocoder = get_vocoder(model_config, device)

    dt_now = datetime.datetime.now()
    run_name = dt_now.strftime('%m:%d:%H:%M')
    run_dir = f"./output/{run_name}/"
    if args.local_rank == 0 and not args.without_save_wav:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(run_dir + "wav_wo_sign/", exist_ok=True)
        os.makedirs(run_dir + "wav_w_sign/", exist_ok=True)
        pred_txt_file = run_dir + "pred.txt"
    if args.local_rank == 0 and args.save_ckpt:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(run_dir + "ckpt", exist_ok=True)

    if args.local_rank == 0 and args.use_wandb:
        wandb.init(
            project='Sign2Speech',
            group="weak prosody reconstruction",
            job_type="training",
            name=run_name,
            config={
                "preprocess": preprocess_config,
                "model": model_config,
                "train": train_config
            },
        )

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
    for epoch in range(step_count, total_step + n_epochs_decay):    # outer loop for different epochs.
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        if args.local_rank == 0:
            progress_inner = tqdm(total=len(loader), leave=True, desc=f"EPOCH {epoch}")
        if distributed:
            sampler.set_epoch(epoch)

        for batchs in loader:  # inner loop within one epoch
            for batch in batchs:

                total_iters += len(batch["sign"].raw_texts)
                model.set_input(batch)         # unpack data from dataset and apply preprocessing
                loss_log = model.optimize_parameters()   # calculate loss functions, get gradients, update network weights


                if args.local_rank == 0 and total_iters % log_step == 0:    # print training losses and save logging information to the disk
                    log = { "epoch": epoch }
                    lr_dict = model.get_learning_rate()
                    log.update(lr_dict)
                    log.update(loss_log)

                if args.local_rank == 0 and total_iters % synth_step == 0 and not args.without_save_wav:
                    ### Save Audio conditioned by text and sign
                    output = model.fake_audio_with_sign
                    output_lens = model.fake_audio_with_sign_lens
                    raw_text = model.fake_raw_texts[0]
                    video_name = model.real_sign.video_names[0]
                    mel_len = output_lens[0].item()
                    if mel_len == 0:
                        print("0 length mel occured")
                        print(raw_text)
                    else:
                        mel_prediction = output[0, :, :mel_len].detach().transpose(1, 2)
                        wav_prediction = vocoder_infer(
                            mel_prediction,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )[0]
                        sf.write(
                            run_dir + f"wav_w_sign/synth_{total_iters // synth_step}.wav",
                            wav_prediction, samplerate=sampling_rate)
                        with open(pred_txt_file, "a") as f:
                            f.write(f"{video_name} | {raw_text}\n")

                    ### Save Audio conditioned by only text
                    output = model.synth_audio
                    output_lens = model.synth_audio_lens
                    mel_len = output_lens[0].item()
                    if mel_len == 0:
                        print("0 length mel occured")
                        print(raw_text)
                    else:
                        mel_prediction = output[0, :, :mel_len].detach().transpose(1, 2)
                        wav_prediction = vocoder_infer(
                            mel_prediction,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )[0]
                        sf.write(
                            run_dir + f"wav_wo_sign/synth_{total_iters // synth_step}.wav",
                            wav_prediction, samplerate=sampling_rate)
                    del output, output_lens, mel_len, mel_prediction, wav_prediction, raw_text
                    torch.cuda.empty_cache()

                if args.local_rank == 0 and args.use_wandb and total_iters % log_step == 0:
                    wandb.log(log)

            if args.local_rank == 0:
                progress_inner.update()
        if args.local_rank == 0:
            progress.update()
        if args.local_rank == 0 and args.save_ckpt and (epoch + 1) % save_epochs == 0:
            print(f"saving the latest model {epoch=}")
            ckpt_save_path = run_dir + f"ckpt/{epoch}.pth"
            model.save_networks(ckpt_save_path)


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngpus', type=int, default=1,
                        help='number of gpus used, equivilent to world_size(local)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='pass rank info throughout the script, DO NOT input through command line')
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
