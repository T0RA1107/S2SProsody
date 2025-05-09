from collections import namedtuple
import torch
import torch.nn as nn

from libs.util.tool import get_mask_from_lengths
from .modules.transformer.Models import S2SMixer, MoE
from .modules import VarianceAdaptorWithReference
from .fastspeech2 import FastSpeech2
from .visual_backbone import PartedPoseBackbone

SpeechPrediction = namedtuple("SpeechPrediction", [
    "mels_wo_sign",
    "p_predictions_wo_sign",
    "e_predictions_wo_sign",
    "mels_w_sign",
    "p_predictions_w_sign",
    "e_predictions_w_sign",
    "p_predictions",
    "e_predictions",
    "log_d_predictions",
    "d_rounded",
    "src_masks",
    "mel_masks",
    "src_lens",
    "mel_lens",
    "weight_sign",
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
        self.MoE = MoE(model_config)

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        key_point=None  # [B, C, T, V]: Batch, Channels, Time, VisualKeypoints
    ) -> SpeechPrediction:
        if len(max_src_len.shape) > 0:
            max_src_len = max_src_len[0]
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        phoneme_embedding = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            phoneme_embedding = phoneme_embedding + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Without Sign Path
        with torch.no_grad():
            (
                output_wo_sign,
                p_predictions_wo_sign,
                e_predictions_wo_sign,
                log_d_predictions,
                d_rounded,
                mel_lens,
                mel_masks,
            ) = self.variance_adaptor(
                phoneme_embedding,
                phoneme_embedding,
                src_masks,
                p_control=p_control,
                e_control=e_control,
                d_control=d_control,
            )

            output_wo_sign, mel_masks = self.decoder(output_wo_sign, mel_masks)
            mels_wo_sign = self.mel_linear(output_wo_sign)

            mels_wo_sign = self.postnet(mels_wo_sign) + mels_wo_sign

        # With Sign Path
        mels_w_sign = p_predictions_w_sign = e_predictions_w_sign = weight_sign = None
        if key_point is not None:
            sign_embbeding = self.sign_processer(key_point)  # [B, C, T, V] -> [B, T, C]
            sign_embbeding = self.visual_project(sign_embbeding)
            prosody_embedding = self.s2s_mixier(phoneme_embedding, sign_embbeding)

            weight_sign = torch.sigmoid(self.MoE(phoneme_embedding, sign_embbeding))

            (
                _, p_predictions_w_sign, e_predictions_w_sign, *_
            ) = self.variance_adaptor(
                phoneme_embedding,
                prosody_embedding,
                src_masks,
                duration_target=d_rounded,
                p_control=p_control,
                e_control=e_control,
                d_control=d_control,
                only_prediction=True
            )
            p_predictions = weight_sign[:, 0, :] * p_predictions_w_sign + (1 - weight_sign[:, 0, :]) * p_predictions_wo_sign
            e_predictions = weight_sign[:, 0, :] * e_predictions_w_sign + (1 - weight_sign[:, 0, :]) * e_predictions_wo_sign
            # print(weight_sign.shape)
            # print(p_predictions_w_sign.shape, e_predictions_w_sign.shape)
            # print(p_predictions.shape, e_predictions.shape)
            (
                output_w_sign, *_
            ) = self.variance_adaptor(
                phoneme_embedding,
                None,
                src_masks,
                duration_target=d_rounded,
                pitch_target=p_predictions,
                energy_target=e_predictions,
                p_control=p_control,
                e_control=e_control,
                d_control=d_control,
            )
            output_w_sign, _ = self.decoder(output_w_sign, mel_masks)
            mels_w_sign = self.mel_linear(output_w_sign)

            mels_w_sign = self.postnet(mels_w_sign) + mels_w_sign

        return SpeechPrediction(
            mels_wo_sign,
            p_predictions_wo_sign,
            e_predictions_wo_sign,
            mels_w_sign,
            p_predictions_w_sign,
            e_predictions_w_sign,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            weight_sign,
        )
