from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from libs.models import SemiCycleGANModelWrapper
from .model import vocoder_infer
from .tool import expand, calc_normed_params


def plot_mel(data: Tuple[npt.NDArray[np.float32], ...], stats: Tuple[float, ...], name: str):
    fig, axd = plt.subplot_mosaic([["mel"], ["mel"], ["weight"]])
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    # mel, pitch, energy, weight_pitch, weight_energy = data
    mel, pitch, energy, weight = data
    pitch = pitch * pitch_std + pitch_mean
    ax_mel = axd["mel"]
    ax_mel.imshow(mel, origin="lower")
    ax_mel.set_aspect(2.5, adjustable="box")
    ax_mel.set_ylim(0, mel.shape[0])
    ax_mel.tick_params(labelsize="x-small", left=False, labelleft=False)
    ax_mel.set_anchor("W")

    ax1 = add_axis(fig, ax_mel)
    ax1.plot(pitch, color="tomato")
    ax1.set_xlim(0, mel.shape[1])
    ax1.set_ylim(0, pitch_max)
    ax1.set_ylabel("F0", color="tomato", fontsize="x-large")
    ax1.tick_params(
        labelsize="x-small", colors="tomato", bottom=False, labelbottom=False)

    ax2 = add_axis(fig, ax_mel)
    ax2.plot(energy, color="darkviolet")
    ax2.set_xlim(0, mel.shape[1])
    ax2.set_ylim(energy_min, energy_max)
    ax2.set_ylabel("Energy", color="darkviolet", fontsize="x-large")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(
        labelsize="x-small", colors="darkviolet",
        bottom=False, labelbottom=False,
        left=False, labelleft=False,
        right=True, labelright=True,
    )

    pos_weight = ax_mel.get_position()
    pos_weight.y1, pos_weight.y0 = pos_weight.y0 - 0.05, pos_weight.y0 - (pos_weight.y1 - pos_weight.y0) * 0.5  # type: ignore[misc]
    ax_weight = axd["weight"]
    ax_weight.set_position(pos_weight)
    ax_weight.plot(weight, color="tomato")
    ax_weight.set_xlim(0, mel.shape[1])
    ax_weight.set_ylim(0, 1)
    ax_weight.set_ylabel("Weight", color="tomato")
    ax_weight.tick_params(
        labelsize="x-small", colors="tomato", bottom=False, labelbottom=False)
    ax_weight.set_anchor("W")

    fig.savefig(f"{name}.png")
    plt.close()


def plot_sign_prosody_dist(
    sign_prosody_label: npt.NDArray[np.float32],
    sign_prosody_predictions: npt.NDArray[np.float32],
    energy_mean: float, pitch_mean: float, save_dir: Path, name: str):
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
        fig.savefig(save_dir / f"{label}|{name}.png")
        plt.close()


def save_inference(model_wrapper: SemiCycleGANModelWrapper, vocoder, data_loader,
                   local_rank: int, sampling_rate: int, stats: Tuple[float, ...],
                   output_dir: Path, epoch: int, model_config: Dict[str, Any], preprocess_config: Dict[str, Any],
                   vid2gender: Dict[str, str], need_prosody_dist: bool=True):
    model_wrapper.model.set_eval_mode()
    save_wav_dir = output_dir / "wavs" / str(epoch)
    save_dist_dir = output_dir / "dists" / str(epoch)
    save_mel_dir = output_dir / "mels" / str(epoch)
    save_metadata_dir = output_dir / "metadata"
    save_wav_dir.mkdir(exist_ok=True, parents=True)
    save_mel_dir.mkdir(exist_ok=True, parents=True)
    save_dist_dir.mkdir(exist_ok=True, parents=True)
    save_metadata_dir.mkdir(exist_ok=True, parents=True)
    name_list = []
    text_list = []
    l2_list = []
    sign_info = model_wrapper.model.sign_info
    for batchs in data_loader:
        for batch in batchs:
            output_wo_sign, output_w_sign, speakers = model_wrapper.inference(batch, vid2gender, estimateProsody=need_prosody_dist)
            energy_mean, pitch_mean = calc_normed_params(output_w_sign, model_wrapper.model.speaker_info, speakers)
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
                txt_id = batch.raw_texts[i]#id(batch.raw_texts[i])
                sf.write(save_wav_dir / f"wo|{txt_id}.wav", wav_prediction, samplerate=sampling_rate)
                duration = output_wo_sign.d_rounded[i]
                plot_mel((
                    mel_prediction_wo_sign.squeeze(0).cpu().numpy(),
                    expand(output_wo_sign.p_predictions_mixed[i], duration),
                    expand(output_wo_sign.e_predictions_mixed[i], duration),
                    expand(output_wo_sign.weight_sign[i][0].cpu().numpy(), duration)),
                    stats, (save_mel_dir / f"wo|{txt_id}").as_posix())
                # with sign
                mel_len_w_sign = output_w_sign.mel_lens[i]
                mel_prediction_w_sign = output_w_sign.mels[i, :, :mel_len_w_sign].detach().transpose(1, 2)
                wav_prediction = vocoder_infer(
                    mel_prediction_w_sign,
                    vocoder,
                    model_config,
                    preprocess_config,
                )[0]
                sf.write(save_wav_dir / f"w|{txt_id}.wav", wav_prediction, samplerate=sampling_rate)
                duration = output_w_sign.d_rounded[i]
                plot_mel((
                    mel_prediction_w_sign.squeeze(0).cpu().numpy(),
                    expand(output_w_sign.p_predictions_mixed[i], duration),
                    expand(output_w_sign.e_predictions_mixed[i], duration),
                    expand(output_w_sign.weight_sign[i][0].cpu().numpy(), duration)),
                    stats, (save_mel_dir / f"w|{txt_id}").as_posix())
                if need_prosody_dist:
                    plot_sign_prosody_dist(
                        batch.prosody_label[i].numpy(), output_w_sign.sign_prosody_predictions[i],
                        energy_mean[i] * sign_info["std"][0] + sign_info["mean"][0], pitch_mean[i] * sign_info["std"][1] + sign_info["mean"][1],
                        save_dist_dir, txt_id)
                # add metadata
                name_list.append(batch.video_names[i])
                text_list.append(txt_id)
                max_len = max(mel_len_wo_sign, mel_len_w_sign)
                l2 = F.mse_loss(
                    F.pad(mel_prediction_wo_sign, (0, max_len - mel_len_wo_sign)),
                    F.pad(mel_prediction_w_sign,  (0, max_len - mel_len_w_sign))).detach().cpu().numpy()
                l2_list.append(l2)

            torch.cuda.empty_cache()
    metadata = pd.DataFrame({ "name": name_list, "text": text_list, "L2": l2_list })
    metadata.to_csv(save_metadata_dir / f"{epoch}_{local_rank}.tsv", sep="\t", index=False)


def calc_expressiveness(model_wrapper: SemiCycleGANModelWrapper, data_loader, stats: Tuple[float, ...]):
    model_wrapper.model.set_eval_mode()
    import json
    with open("../data/VCTK-Corpus/preprocess/stats.json") as f:
        raw_stats = json.load(f)
    pitch_min, pitch_max, pitch_mean, pitch_std = raw_stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = raw_stats["energy"]
    pitch_expressiveness_w_sign = []
    energy_expressiveness_w_sign = []
    pitch_expressiveness_wo_sign = []
    energy_expressiveness_wo_sign = []
    for batchs in data_loader:
        for batch in batchs:
            output_wo_sign, output_w_sign, speakers = model_wrapper.inference(batch)
            bs = output_wo_sign.mels.shape[0]
            for i in range(bs):
                pitch_std_w_sign = expand(output_w_sign.p_predictions_mixed[i], output_w_sign.d_rounded[i]).std()
                energy_std_w_sign = expand(output_w_sign.e_predictions_mixed[i], output_w_sign.d_rounded[i]).std()
                pitch_expressiveness_w_sign.append(pitch_std_w_sign)
                energy_expressiveness_w_sign.append(energy_std_w_sign)
                pitch_std_wo_sign = expand(output_wo_sign.p_predictions_mixed[i], output_wo_sign.d_rounded[i]).std()
                energy_std_wo_sign = expand(output_wo_sign.e_predictions_mixed[i], output_wo_sign.d_rounded[i]).std()
                pitch_expressiveness_wo_sign.append(pitch_std_wo_sign)
                energy_expressiveness_wo_sign.append(energy_std_wo_sign)
    pitch_expressiveness_w_sign = pitch_std * np.mean(pitch_expressiveness_w_sign)
    energy_expressiveness_w_sign = energy_std * np.mean(energy_expressiveness_w_sign)
    pitch_expressiveness_wo_sign = pitch_std * np.mean(pitch_expressiveness_wo_sign)
    energy_expressiveness_wo_sign = energy_std * np.mean(energy_expressiveness_wo_sign)
    print(f"Pitch expressiveness w/  sign: {np.mean(pitch_expressiveness_w_sign):.4f}")
    print(f"Energy expressiveness w/  sign: {np.mean(energy_expressiveness_w_sign):.4f}")
    print(f"Pitch expressiveness w/o sign: {np.mean(pitch_expressiveness_wo_sign):.4f}")
    print(f"Energy expressiveness w/o sign: {np.mean(energy_expressiveness_wo_sign):.4f}")


def save_metadata(n_gpus: int, output_dir: Path, epoch: int):
    save_metadata_dir = output_dir / "metadata"
    save_metadata_dir.mkdir(exist_ok=True)
    metadata = pd.concat(
        pd.read_csv(save_metadata_dir / f"{epoch}_{local_rank}.tsv", sep="\t") for local_rank in range(n_gpus)
    ).drop_duplicates(subset="name").reset_index(drop=True)
    for local_rank in range(n_gpus):
        (save_metadata_dir / f"{epoch}_{local_rank}.tsv").unlink()
    l2_list = metadata["L2"]
    arg_idx = len(l2_list) - np.argsort(np.argsort(l2_list))
    metadata["L2_index"] = arg_idx
    metadata.to_csv(save_metadata_dir / f"{epoch}.tsv", sep="\t", index=False)


def save_validation_loss(model_wrapper: SemiCycleGANModelWrapper, data_loader: Any):
    valid_loss_logs: dict[str, list] = {
        "GAN loss/G (valid)": [],
        "prosody loss/v_loss (valid)": [],
        "prosody loss/a_loss (valid)": [],
        "prosody loss/total (valid)": [],
        "Regularization/Energy mean (valid)": [],
        "Regularization/Pitch mean (valid)": [],
        # Output memo
        "output/pitch std (without sign; valid)": [],
        "output/energy std (without sign; valid)": [],
        "output/pitch std (with sign; valid)": [],
        "output/energy std (with sign; valid)": [],
    }
    for batchs in data_loader:
        for batch in batchs:
            loss_log = model_wrapper.validate(batch)
            for key, val in loss_log.items():
                valid_loss_logs[key].append(val)

    return { key: torch.tensor(val).mean() for key, val in valid_loss_logs.items() }
