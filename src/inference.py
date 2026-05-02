import argparse
from pathlib import Path
import yaml
import json

import torch
from torch.utils.data import DataLoader

# from FastSpeech2.evaluate import evaluate
from libs.util.model import get_vocoder

from libs.models import SemiCycleGANModel, SemiCycleGANModelWrapper
from dataset.dataset import AudioDataset, SignDataset
from libs.util.save_data import save_inference, calc_expressiveness


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
        phase="test", split="test", partial_list_path=args.partial_list_path
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
        configs_ft=configs_ft, isTrain=True, distributed=False)      # create a model given opt.model and other options
    model.load_networks(args.ckpt_path)
    model.setup(train_config)               # regular setup: load and print networks; create schedulers
    model_wrapper = SemiCycleGANModelWrapper(model)
    vocoder = get_vocoder(model_config, device)

    output_dir = Path(args.output_dir)
    if not args.without_save_wav:
        wav_dir = output_dir / "wavs"
        if args.local_rank == 0:
            output_dir.mkdir(exist_ok=True)
            wav_dir.mkdir(exist_ok=True)

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    calc_expressiveness(model_wrapper, inference_loader, stats)

    # save_inference(model_wrapper, vocoder, inference_loader,
    #                args.local_rank, sampling_rate, stats, wav_dir,
    #                -1, model_config, preprocess_config, inference_dataset.partial_vid2gender,
    #                need_prosody_dist=True)

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
        "--output_dir", type=str, required=True, help="path to output directory"
    )
    parser.add_argument(
        "--save_ckpt", action="store_true"
    )
    parser.add_argument(
        "--without_save_wav", action="store_true"
    )
    parser.add_argument(
        "--partial_list_path", type=str, default=None,
        help="path to a text file listing video IDs to run inference on (default: full test set)"
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
