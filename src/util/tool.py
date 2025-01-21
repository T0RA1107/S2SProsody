import os
import random

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.nn.functional as F

from FastSpeech2.utils.model import vocoder_infer

from models.semi_cycle_gan import SemiCycleGANModel


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


def save_inference(model: SemiCycleGANModel, vocoder, data_loader, sampling_rate, stats, output_dir, epoch, model_config, preprocess_config):
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
