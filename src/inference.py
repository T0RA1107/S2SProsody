import argparse
from pathlib import Path
import datetime
import yaml
import json

import torch
from torch.utils.data import DataLoader

# from FastSpeech2.evaluate import evaluate
from libs.util.model import get_vocoder

from libs.models.semi_cycle_gan import SemiCycleGANModel
from dataset.dataset import AudioDataset, SignDataset
from libs.util.save_data import save_inference


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs, configs_ft):
    args.local_rank = 0
    args.ngpus = 1

    preprocess_config, model_config, train_config = configs

    audio_dataset = AudioDataset(
        "train.txt", preprocess_config, train_config, model_config)  # to get speaker info from audio train dataset
    train_dataset = SignDataset(
        preprocess_config, train_config, args.local_rank,
        phase="train", split="train"
    )
    inference_dataset = SignDataset(
        preprocess_config, train_config, args.local_rank,
        phase="test", split="test", partial_list_path="./test_0.txt"
    )
    speaker_info = audio_dataset.get_speaker_info()
    sign_info = train_dataset.get_sign_prosody_info()
    with Path(preprocess_config["path"]["preprocessed_path"], "stats.json").open() as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]


    if "speaker_num" not in model_config:
        model_config["speaker_num"] = audio_dataset.speaker_num
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        collate_fn=inference_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )

    model = SemiCycleGANModel(
        args, preprocess_config, model_config, train_config,
        speaker_info=speaker_info, sign_info=sign_info,
        configs_ft=configs_ft, isTrain=False, distributed=False)      # create a model given opt.model and other options
    model.load_networks(args.ckpt_path)
    model.setup(train_config)               # regular setup: load and print networks; create schedulers

    vocoder = get_vocoder(model_config, device)

    dt_now = datetime.datetime.now()
    run_name = dt_now.strftime("%m:%d:%H:%M")
    output_dir = Path(train_config["path"]["output_path"], run_name)
    if not args.without_save_wav:
        wav_dir = output_dir / "wavs"
        if args.local_rank == 0:
            output_dir.mkdir(exist_ok=True)
            wav_dir.mkdir(exist_ok=True)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    save_inference(model, vocoder, inference_loader,
                   args.local_rank, sampling_rate, stats, wav_dir,
                   -1, model_config, preprocess_config, inference_dataset.partial_vid2gender,
                   need_prosody_dist=False)

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
        "--ckpt_path", type=str, required=True
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--save_ckpt", action="store_true"
    )
    parser.add_argument(
        "--without_save_wav", action="store_true"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs, None)
