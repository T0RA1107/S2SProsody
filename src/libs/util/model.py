import os
import json

import torch
import torch.nn as nn
import numpy as np

from libs.models import hifigan
from libs.models import FastSpeech2
from libs.models.optimizer import ScheduledOptim


def get_model(args, configs, device, configs_ft=None, train=False):
    (preprocess_config, model_config, train_config) = configs

    if configs_ft is None:
        model = FastSpeech2(preprocess_config, model_config).to(device)
    else:
        (preprocess_config_ft, model_config_ft, _) = configs_ft
        model = FastSpeech2(preprocess_config_ft, model_config_ft).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        print(ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        elif configs_ft is not None:
            assert args.restore_step_ft is not None
            train_config_ft = configs_ft[2]
            ckpt_path = os.path.join(
                train_config_ft["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step_ft),
            )
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            if model_config["multi_speaker"]:
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                model.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                ).to(device)
                emb_params = []
                other_params = []
                for name, param in model.named_parameters():
                    if name == "speaker_emb.weight":
                        emb_params.append(param)
                    else:
                        other_params.append(param)
                if isinstance(train_config["optimizer"]["lr"], float):
                    _optimizer = torch.optim.Adam(
                        [{ "params": emb_params,
                        "lr": train_config["optimizer"]["lr_fine-tuning"],
                        "init_lr": train_config["optimizer"]["lr_fine-tuning"] },
                        { "params": other_params,
                        "lr": train_config["optimizer"]["lr"],
                        "init_lr": train_config["optimizer"]["lr"] },
                        ],
                        betas=train_config["optimizer"]["betas"],
                        eps=train_config["optimizer"]["eps"],
                        weight_decay=train_config["optimizer"]["weight_decay"],
                    )
                    scheduled_optim._optimizer = _optimizer

        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    hifigan_path = config["vocoder"]["hifigan_path"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open(f"{hifigan_path}/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(f"{hifigan_path}/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load(f"{hifigan_path}/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
