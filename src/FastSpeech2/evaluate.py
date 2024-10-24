import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from FastSpeech2.utils.model import get_model, get_vocoder
from FastSpeech2.utils.tools import to_device, log_at_wandb, synth_one_sample
from FastSpeech2.model import FastSpeech2Loss
from FastSpeech2.dataset import Dataset
from FastSpeech2.metrics import calc_pitch_moment, calc_pitch_dtw


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, log=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if log is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log_at_wandb(log,  losses=loss_means, phase="Valid")
        log_at_wandb(
            log,
            fig=fig,
            tag="Valid/{}".format(tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log_at_wandb(
            log,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Valid/{}_reconstructed".format(tag),
        )
        log_at_wandb(
            log,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Valid/{}_synthesized".format(tag),
        )

    return message


def evaluate_prosody(model, step, configs, log=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Evaluation
    moment_abs_error = [0. for _ in range(3)]
    dtw_sum = 0
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():

                # Forward
                output = model(*(batch[2:]))

                # Cal Pitch Moment and Pitch DTW
                pitch_prediction = output[2].detach().cpu()
                pitch_target = batch[9].detach().cpu()

                moments = calc_pitch_moment(pitch_prediction, pitch_target)

                for i in range(3):
                    moment_abs_error[i] += (moments[i][0] - moments[i][1]).abs().sum().item()

                dtw = calc_pitch_dtw(pitch_prediction, pitch_target)
                dtw_sum += dtw * len(batch[0])

    moments_mean_abs_error = [moment_sum / len(dataset) for moment_sum in moment_abs_error]
    dtw_mean = dtw_sum / len(dataset)

    message = "Validation Step {}, σ MAE: {:.4f}, γ MAE: {:.4f}, Κ MAE: {:.4f}, DTW mean: {:.4f}".format(
        *([step] + moments_mean_abs_error + [dtw_mean])
    )

    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
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
        "--fine_tuning", action="store_true"
    )
    parser.add_argument(
        "--prosody", action="store_true"
    )
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

    # Get model
    # model = get_model(args, configs, device, train=False).to(device)
    model = get_model(args, configs, device, configs_ft=configs_ft, train=False).to(device)

    if not args.prosody:
        message = evaluate(model, args.restore_step, configs)
    else:
        message = evaluate_prosody(model, args.restore_step, configs)
    print(message)
