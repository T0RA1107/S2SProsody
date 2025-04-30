import random
from typing import Dict
import torch
import torch.nn.functional as F

from datasets.sign_dataset import SignData
from .semi_cycle_gan import SemiCycleGANModel, TrainOutput, InferenceOutput
from .semi_cycle_gan import random_clip_batch


class SemiCycleGANModelWrapper:
    def __init__(self, model: SemiCycleGANModel):
        self.model: SemiCycleGANModel = model

    @torch.no_grad()
    def validate(self, sign: SignData):
        batch_size = sign.text_tokens.shape[0]
        token_length = sign.token_length.to(self.model.device, non_blocking=True)
        max_src_len = token_length.max().to(self.model.device, non_blocking=True)
        text_tokens = sign.text_tokens[:, :max_src_len].to(self.model.device, non_blocking=True)

        speakers = random.choices(self.model.all_speakers, k=batch_size)
        speakers = torch.tensor(speakers, device=self.model.device).long()

        # with sign language TTS
        pred = self.model.netG_sign2audio(
            speakers,
            text_tokens,
            token_length,
            max_src_len,
            key_point=sign.visual_prefix.to(self.model.device, non_blocking=True)
        )

        audio = pred.mels_w_sign.masked_fill(
            pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.mels_w_sign.shape[2]), 0.0).unsqueeze(1)
        audio_lens = pred.mel_lens

        prosody_predictions = torch.cat([
            pred.p_predictions.unsqueeze(1),
            pred.e_predictions.unsqueeze(1),
            pred.weight_sign
        ], dim=1)
        sign_prosody_predictions = self.model.netProsody_estimator(prosody_predictions)
        output_w_sign = TrainOutput(
            audio, sign.token_length, audio_lens, pred.src_masks, pred.mel_masks,
            pred.p_predictions, pred.e_predictions, pred.log_d_predictions, pred.d_rounded,
            sign_prosody_predictions, pred.weight_sign.detach().cpu())

        # GAN Loss
        fake = output_w_sign.mels
        clip_length = output_w_sign.mel_lens.detach().min().cpu() + 8
        fake = random_clip_batch(fake, output_w_sign.mel_lens, clip_length)

        pred_fake = self.model.netD_audio(fake)
        loss_G_audio = self.model.criterionGAN(pred_fake, True)

        # Prosody Reconstruction Loss
        prosody_label = sign.prosody_label.to(self.model.device, non_blocking=True)
        _, pr_info = self.model.criterionPR(sign_prosody_predictions, prosody_label)

        # Prosody Guided Regularization Loss
        _, pgr_info = self.model.criterionRGR(output_w_sign, prosody_label, speakers.detach().cpu().tolist())

        loss_log = {
            "GAN loss/G (valid)": loss_G_audio.detach().cpu().item(),
            "prosody loss/v_loss (valid)": pr_info.v_loss,
            "prosody loss/a_loss (valid)": pr_info.a_loss,
            "prosody loss/total (valid)": pr_info.total,
            "Regularization/Energy mean (valid)": pgr_info.energy,
            "Regularization/Pitch mean (valid)": pgr_info.pitch,
        }

        torch.cuda.empty_cache()
        return loss_log

    @torch.no_grad()
    def inference(self, sign: SignData, vid2gender: Dict[str, str]=None, estimateProsody: bool=False):
        token_length = sign.token_length.to(self.model.device, non_blocking=True)
        max_src_len = token_length.max().to(self.model.device, non_blocking=True)
        text_tokens = sign.text_tokens[:, :max_src_len].to(self.model.device, non_blocking=True)

        if vid2gender is not None:
            speakers = [self.model.target_speakers[vid2gender[vid]] for vid in sign.video_names]
        else:
            speakers = [self.model.target_speakers[["male", "female"][random.randint(0, 1)]] for _ in range(len(sign.video_names))]
        speakers = torch.tensor(speakers, device=self.model.device).long()

        # without sign language TTS
        pred = self.model.netG_sign2audio(
            speakers,
            text_tokens,
            token_length,
            max_src_len,
            key_point=sign.visual_prefix.to(self.model.device, non_blocking=True)
        )

        audio_wo_sign = pred.mels_wo_sign.masked_fill(
            pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.mels_wo_sign.shape[2]), 0.0).unsqueeze(1)

        audio_wo_sign_lens = pred.mel_lens.detach().cpu()

        pred_prosody_label = None
        if estimateProsody:
            prosody_predictions = torch.cat([
                pred.p_predictions_wo_sign.unsqueeze(1),
                pred.e_predictions_wo_sign.unsqueeze(1),
                torch.zeros_like(pred.p_predictions_wo_sign.unsqueeze(1), device=self.model.device).repeat((1, 2, 1))
            ], dim=1)
            pred_prosody_label = self.model.netProsody_estimator(prosody_predictions)
            pred_prosody_label = F.softmax(pred_prosody_label, dim=-1).cpu().numpy()

        output_wo_sign = InferenceOutput(
            audio_wo_sign, sign.token_length.numpy(), audio_wo_sign_lens,
            pred.p_predictions_wo_sign.cpu().numpy(), pred.e_predictions_wo_sign.cpu().numpy(), pred.d_rounded.cpu().numpy(),
            pred_prosody_label)

        # with sign language TTS
        audio_w_sign = pred.mels_w_sign.masked_fill(
            pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.mels_w_sign.shape[2]), 0.0).unsqueeze(1)
        audio_w_sign_lens = pred.mel_lens

        pred_prosody_label = None
        if estimateProsody:
            prosody_predictions = torch.cat([
                pred.p_predictions_w_sign.unsqueeze(1),
                pred.e_predictions_w_sign.unsqueeze(1),
                pred.weight_sign
            ], dim=1)
            pred_prosody_label = self.model.netProsody_estimator(prosody_predictions)
            pred_prosody_label = F.softmax(pred_prosody_label, dim=-1).cpu().numpy()

        output_w_sign = InferenceOutput(
            audio_w_sign, sign.token_length.numpy(), audio_w_sign_lens,
            pred.p_predictions_w_sign.cpu().numpy(), pred.e_predictions_w_sign.cpu().numpy(), pred.d_rounded.cpu().numpy(),
            pred_prosody_label)

        torch.cuda.empty_cache()
        return output_wo_sign, output_w_sign, speakers.cpu().numpy()

