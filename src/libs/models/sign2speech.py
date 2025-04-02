from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.util.tool import get_mask_from_lengths
from .modules.transformer.Models import S2SMixer
from .modules import VarianceAdaptorWithReference
from .fastspeech2 import FastSpeech2
from .visual_backbone import PartedPoseBackbone

SpeechPrediction = namedtuple("SpeechPrediction", [
    "output",
    "postnet_output",
    "p_predictions",
    "e_predictions",
    "log_d_predictions",
    "d_rounded",
    "src_masks",
    "mel_masks",
    "src_lens",
    "mel_lens",
    "weight_sign",
    "pitch_confidence",
    "energy_confidence",
    "log_duration_confidence"
])


class Sign2Speech(FastSpeech2):

    def __init__(self, preprocess_config, model_config):
        super(Sign2Speech, self).__init__(
            preprocess_config,
            model_config
        )
        self.sign_processer = PartedPoseBackbone()
        self.s2s_mixier = S2SMixer(model_config)
        for name, param in self.s2s_mixier.named_parameters():
            nn.init.zeros_(param)
        self.max_seq_len = model_config["max_seq_len"]
        self.variance_adaptor = VarianceAdaptorWithReference(self.variance_adaptor)
        self.dim_visual = self.sign_processer.feat_dim
        self.dim_embedding = self.s2s_mixier.d_model
        if self.dim_visual != self.dim_embedding:
            self.visual_project = nn.Linear(self.dim_visual, self.dim_embedding)
        else:
            self.visual_project == nn.Identity()
        self.MoE = nn.Sequential(
            nn.Linear(2 * self.dim_embedding, self.dim_embedding),
            nn.ReLU(),
            nn.Linear(self.dim_embedding, 1)
        )

        self.speaker_text_embedding = None

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        key_point=None  # [B, C, T, V]: Batch, Channels, Time, VisualKeypoints
    ) -> SpeechPrediction:
        if len(max_src_len.shape) > 0:
            max_src_len = max_src_len[0]
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        weight_sign = None
        # Cross Attention with keypoint
        if key_point is not None:
            sign_embbeding = self.sign_processer(key_point)  # [B, C, T, V] -> [B, T, C]
            sign_embbeding = self.visual_project(sign_embbeding)
            output_crsattn = self.s2s_mixier(output, sign_embbeding)
            concat_prosody_embedding = torch.cat((output_crsattn.mean(dim=1), output.mean(dim=1)), dim=1)
            weight_sign = torch.sigmoid(self.MoE(concat_prosody_embedding)).unsqueeze(2)
            prosody_embedding = weight_sign * output_crsattn + (1 - weight_sign) * output

            (
                output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                mel_lens,
                mel_masks,
                pitch_confidence,
                energy_confidence,
                log_duration_confidence
            ) = self.variance_adaptor(
                output,
                prosody_embedding,
                src_masks,
                mel_masks,
                max_mel_len,
                p_targets,
                e_targets,
                d_targets,
                p_control,
                e_control,
                d_control,
            )
        else:
            (
                output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                mel_lens,
                mel_masks,
                pitch_confidence,
                energy_confidence,
                log_duration_confidence
            ) = self.variance_adaptor(
                output,
                output,
                src_masks,
                mel_masks,
                max_mel_len,
                p_targets,
                e_targets,
                d_targets,
                p_control,
                e_control,
                d_control,
            )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return SpeechPrediction(
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            weight_sign,
            pitch_confidence,
            energy_confidence,
            log_duration_confidence
        )
