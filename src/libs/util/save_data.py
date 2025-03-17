import os

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.nn.functional as F

from libs.models import SemiCycleGANModel
from .model import vocoder_infer
from .tool import expand, calc_normed_params


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
    fig.savefig(f"{name}.png")
    plt.close()


def plot_sign_prosody_dist(sign_prosody_label, sign_prosody_predictions, energy_mean, pitch_mean, save_dir, name):
    label_type = ["v_hand", "v_face", "a_hand", "a_face"]
    n = sign_prosody_label.shape[-1]
    bins = np.linspace(0, 1, n + 1)
    x = (bins[:-1] + bins[1:]) / 2.
    mean_GT = (x * sign_prosody_label).sum(1)
    mean_pred = (x * sign_prosody_predictions).sum(1)
    assert np.all(np.abs(sign_prosody_predictions.sum(1) - 1) <= 1e-5), f"{sign_prosody_predictions.shape}, {sign_prosody_predictions.sum(1)}"

    for i, label in enumerate(label_type):
        fig = plt.figure()

        plt.bar(x=x, height=sign_prosody_label[i], width=1 / n,
                color="red", alpha=0.5, label="GT")
        plt.bar(x=x, height=sign_prosody_predictions[i], width=1 / n,
                color="blue", alpha=0.5, label="Prediction")
        plt.axvline(x=mean_GT[i], color="red", linestyle="-", label="mean (GT)")
        plt.axvline(x=mean_pred[i], color="blue", linestyle="--", label="mean (Prediction)")
        if label == "v_hand":
            plt.axvline(x=energy_mean, color="green", linestyle=":", label="Energy mean")
        if label == "v_face":
            plt.axvline(x=pitch_mean, color="green", linestyle=":", label="Pitch mean")
        plt.legend()
        fig.savefig(os.path.join(save_dir, f"{label}|{name}.png"))
        plt.close()


def save_inference(model: SemiCycleGANModel, vocoder, data_loader,
                   local_rank, sampling_rate, stats,
                   output_dir, epoch, model_config, preprocess_config, vid2gender,
                   need_prosody_dist=True):
    model.set_eval_mode()
    save_wav_dir = os.path.join(output_dir, "wavs", str(epoch))
    save_dist_dir = os.path.join(output_dir, "dists", str(epoch))
    save_mel_dir = os.path.join(output_dir, "mels", str(epoch))
    save_metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(save_wav_dir, exist_ok=True)
    os.makedirs(save_mel_dir, exist_ok=True)
    os.makedirs(save_dist_dir, exist_ok=True)
    os.makedirs(save_metadata_dir, exist_ok=True)
    name_list = []
    text_list = []
    l2_list = []
    sign_info = model.sign_info
    for batchs in data_loader:
        for batch in batchs:
            output_wo_sign, output_w_sign, speakers = model.inference(batch, vid2gender, estimateProsody=need_prosody_dist)
            energy_mean, pitch_mean = calc_normed_params(output_w_sign, model.speaker_info, speakers)
            bs = output_wo_sign.mels.shape[0]
            for i in range(bs):
                # without sign
                mel_len_wo_sign = output_wo_sign.mel_lens[i]
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
                    stats, os.path.join(save_mel_dir, f"wo|{batch.raw_texts[i]}"))
                # with sign
                mel_len_w_sign = output_w_sign.mel_lens[i]
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
                    stats, os.path.join(save_mel_dir, f"w|{batch.raw_texts[i]}"))
                if need_prosody_dist:
                    plot_sign_prosody_dist(
                        batch.prosody_label[i].numpy(), output_w_sign.sign_prosody_predictions[i],
                        energy_mean[i] * sign_info["std"][0] + sign_info["mean"][0], pitch_mean[i] * sign_info["std"][1] + sign_info["mean"][1],
                        save_dist_dir, f"{batch.raw_texts[i]}")
                # add metadata
                name_list.append(batch.video_names[i])
                text_list.append(batch.raw_texts[i])
                max_len = max(mel_len_wo_sign, mel_len_w_sign)
                l2 = F.mse_loss(
                    F.pad(mel_prediction_wo_sign, (0, max_len - mel_len_wo_sign)),
                    F.pad(mel_prediction_w_sign,  (0, max_len - mel_len_w_sign))).detach().cpu().numpy()
                l2_list.append(l2)

            torch.cuda.empty_cache()
    metadata = pd.DataFrame({ "name": name_list, "text": text_list, "L2": l2_list })
    metadata.to_csv(os.path.join(save_metadata_dir, f"{epoch}_{local_rank}.tsv"), sep="\t", index=False)


def save_metadata(n_gpus, output_dir, epoch):
    save_metadata_dir = os.path.join(output_dir, "metadata")
    metadata = pd.concat(
        pd.read_csv(os.path.join(save_metadata_dir, f"{epoch}_{local_rank}.tsv"), sep="\t") for local_rank in range(n_gpus)
    ).drop_duplicates(subset="name").reset_index(drop=True)
    for local_rank in range(n_gpus):
        os.remove(os.path.join(save_metadata_dir, f"{epoch}_{local_rank}.tsv"))
    l2_list = metadata["L2"]
    arg_idx = len(l2_list) - np.argsort(np.argsort(l2_list))
    metadata["L2_index"] = arg_idx
    metadata.to_csv(os.path.join(save_metadata_dir, f"{epoch}.tsv"), sep="\t", index=False)


def save_validation_loss(model: SemiCycleGANModel, data_loader):
    valid_loss_logs = {
        "GAN loss/G (valid)": [],
        "prosody loss/v_loss (valid)": [],
        "prosody loss/a_loss (valid)": [],
        "prosody loss/total (valid)": [],
        "Regularization/Energy mean (valid)": [],
        "Regularization/Pitch mean (valid)": [],
    }
    for batchs in data_loader:
        for batch in batchs:
            loss_log = model.validate(batch)
            for key, val in loss_log.items():
                valid_loss_logs[key].append(val)

    return { key: torch.tensor(val).mean() for key, val in valid_loss_logs.items() }
