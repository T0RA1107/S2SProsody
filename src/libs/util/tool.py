import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils

# from libs.models.semi_cycle_gan import InferenceOutput


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


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def make_mask_from_length(length, maxlen):
    idx = np.arange(maxlen).reshape(1, -1).repeat(length.shape[0], axis=0)
    mask = idx < length.reshape(-1, 1)
    return mask


def calc_normed_params(output_w_sign, speaker_info, speakers):
    energy_mean_info = np.array([speaker_info["energy"]["mean"][i] for i in speakers])
    energy_std_info  = np.array([speaker_info["energy"]["std"][i] for i in speakers])
    pitch_mean_info  = np.array([speaker_info["pitch"]["mean"][i] for i in speakers])
    pitch_std_info   = np.array([speaker_info["pitch"]["std"][i] for i in speakers])

    L = output_w_sign.e_predictions.shape[1]
    src_mask = make_mask_from_length(output_w_sign.src_lens, L)
    energy_mean_info = energy_mean_info[:, None].repeat(L, axis=1)
    energy_std_info = energy_std_info[:, None].repeat(L, axis=1)
    norm_energy = (output_w_sign.e_predictions - energy_mean_info) / (energy_std_info + 1e-5)
    norm_energy = norm_energy * src_mask
    energy_mean = norm_energy.sum(axis=1) / output_w_sign.src_lens
    # energy_var = (norm_energy ** 2).sum(axis=1) / token_length - energy_mean ** 2

    pitch_mean_info = pitch_mean_info[:, None].repeat(L, axis=1)
    pitch_std_info = pitch_std_info[:, None].repeat(L, axis=1)
    norm_pitch = (output_w_sign.p_predictions - pitch_mean_info) / (pitch_std_info + 1e-5)
    norm_pitch = norm_pitch * src_mask
    pitch_mean = norm_pitch.sum(axis=1) / output_w_sign.src_lens
    # pitch_var = (norm_pitch ** 2).sum(axis=1) / token_length - pitch_mean ** 2
    return energy_mean, pitch_mean
