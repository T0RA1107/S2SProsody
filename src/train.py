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
from FastSpeech2.utils.model import get_vocoder, vocoder_infer

from models.semi_cycle_gan import SemiCycleGANModel
from dataset import UnpairedAudioSignDataset, SignDataset

# DDP
import torch.distributed as dist

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


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def plot_mel(data, stats, name):
    fig, ax = plt.subplots()
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    mel, pitch, energy = data
    pitch = pitch * pitch_std + pitch_mean
    ax.imshow(mel, origin="lower")
    ax.set_aspect(2.5, adjustable="box")
    ax.set_ylim(0, mel.shape[0])
    ax.tick_params(labelsize="x-small", left=False, labelleft=False)
    ax.set_anchor("W")

    ax1 = add_axis(fig, ax)
    ax1.plot(pitch, color="tomato")
    ax1.set_xlim(0, mel.shape[1])
    ax1.set_ylim(0, pitch_max)
    ax1.set_ylabel("F0", color="tomato")
    ax1.tick_params(
        labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
    )

    ax2 = add_axis(fig, ax)
    ax2.plot(energy, color="darkviolet")
    ax2.set_xlim(0, mel.shape[1])
    ax2.set_ylim(energy_min, energy_max)
    ax2.set_ylabel("Energy", color="darkviolet")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(
        labelsize="x-small",
        colors="darkviolet",
        bottom=False,
        labelbottom=False,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )
    fig.savefig(name)
    plt.close()


def save_inference(model: SemiCycleGANModel, vocoder, data_loader, sampling_rate, stats, output_dir, epoch):
    save_wav_dir = os.path.join(output_dir, "wavs", str(epoch))
    save_mel_dir = os.path.join(output_dir, "mels", str(epoch))
    save_metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(save_wav_dir, exist_ok=True)
    os.makedirs(save_mel_dir, exist_ok=True)
    os.makedirs(save_metadata_dir, exist_ok=True)
    name_list = []
    text_list = []
    l2_list = []
    for batchs in data_loader:
        for batch in batchs:
            output_wo_sign, output_w_sign = model.inference(batch)
            bs = output_wo_sign.mels.shape[0]
            for i in range(bs):
                # without sign
                mel_len_wo_sign = output_wo_sign.lens[i]
                mel_prediction_wo_sign = output_wo_sign.mels[i, :, :mel_len_wo_sign].detach().transpose(1, 2)
                wav_prediction = vocoder_infer(
                    mel_prediction_wo_sign,
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]
                sf.write(os.path.join(save_wav_dir, f"wo|{batch.raw_texts[i]}.wav"), wav_prediction, samplerate=sampling_rate)
                duration = output_wo_sign.d_rounded[i]
                plot_mel((
                    mel_prediction_wo_sign.squeeze(0).cpu().numpy(),
                    expand(output_wo_sign.p_predictions[i], duration),
                    expand(output_wo_sign.e_predictions[i], duration)),
                    stats, os.path.join(save_mel_dir, f"wo|{batch.raw_texts[i]}.png"))
                # with sign
                mel_len_w_sign = output_w_sign.lens[i]
                mel_prediction_w_sign = output_w_sign.mels[i, :, :mel_len_w_sign].detach().transpose(1, 2)
                wav_prediction = vocoder_infer(
                    mel_prediction_w_sign,
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]
                sf.write(os.path.join(save_wav_dir, f"w|{batch.raw_texts[i]}.wav"), wav_prediction, samplerate=sampling_rate)
                duration = output_w_sign.d_rounded[i]
                plot_mel((
                    mel_prediction_w_sign.squeeze(0).cpu().numpy(),
                    expand(output_w_sign.p_predictions[i], duration),
                    expand(output_w_sign.e_predictions[i], duration)),
                    stats, os.path.join(save_mel_dir, f"w|{batch.raw_texts[i]}.png"))
                # add metadata
                name_list.append(batch.video_names[i])
                text_list.append(batch.raw_texts[i])
                max_len = max(mel_len_wo_sign, mel_len_w_sign)
                l2 = F.mse_loss(
                    F.pad(mel_prediction_wo_sign, (0, max_len - mel_len_wo_sign)),
                    F.pad(mel_prediction_w_sign,  (0, max_len - mel_len_w_sign))).detach().cpu().numpy()
                l2_list.append(l2)

            torch.cuda.empty_cache()
    arg_idx = np.argsort(l2_list)[::-1]
    metadata = pd.DataFrame({ "name": name_list, "text": text_list, "L2": l2_list, "index": arg_idx })
    metadata.to_csv(os.path.join(save_metadata_dir, f"{epoch}.tsv"), sep="\t")

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
        args, preprocess_config, train_config, partial_list_path="./valid_list.txt"
    )
    speaker_info = dataset.audio_dataset.get_speaker_info()
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
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )

    model = SemiCycleGANModel(args, preprocess_config, model_config, train_config, speaker_info=speaker_info, configs_ft=configs_ft, distributed=distributed)      # create a model given opt.model and other options
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
        if args.local_rank == 0 and not args.without_save_wav:
            save_inference(model, vocoder, valid_loader, sampling_rate, stats, run_dir, epoch)
        exit()
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
        if args.local_rank == 0 and not args.without_save_wav:
            save_inference(model, vocoder, valid_loader, sampling_rate, wav_dir, epoch)
        if args.local_rank == 0 and args.save_ckpt and (epoch + 1) % save_epochs == 0:
            print(f"saving the latest model {epoch=}")
            ckpt_save_path = run_dir + f"ckpt/{epoch}.pth"
            model.save_networks(ckpt_save_path)
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
