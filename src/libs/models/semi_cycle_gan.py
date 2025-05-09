from pathlib import Path
from collections import namedtuple
import json
import random
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.util.audio_pool import AudioPool
from .base_model import BaseModel
from . import networks
from .loss_fn.prosody_loss import ProsodyReconstructionLoss, ProsodyGuidedRegularizationLoss, IntonationRegularizationLoss
from .sign2speech import Sign2Speech
from .prosody_estimator import ProsodyDistEstimator1D

logger = getLogger(__name__)

TrainOutput = namedtuple("TrainOutput", [
    "mels",
    "src_lens",
    "mel_lens",
    "src_masks",
    "mel_masks",
    "p_predictions",
    "e_predictions",
    "log_d_predictions",
    "d_rounded",
    "sign_prosody_predictions",
    "weight_sign",
])


def random_clip_batch(inputs: torch.Tensor, lengths: torch.Tensor, clip_length: int) -> torch.Tensor:
    # inputs: (B, C, L, ...)
    L = inputs.shape[2]

    clips = []
    for i in range(inputs.shape[0]):
        true_length = min(L, lengths[i].item())
        if true_length - clip_length > 0:
            start = torch.randint(0, true_length - clip_length + 1, (1,)).item()
        else:
            start = 0
        clip = inputs[i, :, start:start + clip_length]
        clips.append(clip)

    return torch.stack(clips, dim=0)


class SemiCycleGANModel(BaseModel):

    def __init__(self, args, preprocess_config, model_config, train_config,
                 speaker_info, sign_info=None, configs_ft=None, isTrain=True, distributed=False):
        BaseModel.__init__(self, args, preprocess_config, model_config, train_config, isTrain)
        self.prosody_dist = train_config["loss"]["prosody"]["dist"]
        self.speaker_info = speaker_info
        self.sign_info = sign_info

        self.netG_sign2audio = Sign2Speech(preprocess_config, model_config)
        self.netG_sign2audio = networks.init_net(self.netG_sign2audio, args, distributed=distributed)
        self.model_names.append("G_sign2audio")

        if configs_ft is not None:
            train_config_ft = configs_ft[2]
            ckpt_path = Path(
                train_config_ft["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step_ft))
            if args.local_rank == 0:
                logger.info(f"Load {ckpt_path.as_posix()}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # self.netG_sign2audio.load_state_dict(ckpt["model"], strict=False)
            if distributed:
                self.netG_sign2audio.module.load_state_dict(ckpt["model"], strict=False)
            else:
                self.netG_sign2audio.load_state_dict(ckpt["model"], strict=False)

        # speaker info
        with Path(preprocess_config["path"]["preprocessed_path"], "speakers.json").open() as f:
            self.speaker_map = json.load(f)
        self.target_speakers = model_config["speaker"]["target"]
        self.all_speakers = list(speaker_info["energy"]["mean"].keys())

        self.step = 0

        if self.isTrain:
            # define discriminators
            self.netD_audio = networks.define_D(
                1, model_config["D_audio"]["ndf"], "audio",
                model_config["D_audio"]["n_layers_D"], model_config["D_audio"]["norm"],
                model_config["D_audio"]["init_type"], model_config["D_audio"]["init_gain"],
                args, distributed)
            self.model_names.append("D_audio")

            # initialize discriminator loss configuration
            self.loss_log_D = {}
            self.weight_prosody = train_config["loss"]["weight"]["prosody"]
            self.weight_regdist_mean = train_config["loss"]["weight"]["regdist_mean"]
            self.weight_intotation = train_config["loss"]["weight"]["intonation"]
            self.weight_gate = train_config["loss"]["weight"]["gate"]

            # initialize prosody estimator
            self.netProsody_estimator = ProsodyDistEstimator1D(
                3, train_config["loss"]["prosody"]["bins"], 4
            )
            self.netProsody_estimator = networks.init_net(self.netProsody_estimator, args, distributed=distributed)

            self.fake_audio_pool = AudioPool(train_config["GAN"]["pool_size"])  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(train_config["GAN"]["gan_mode"]).to(self.device, non_blocking=True)  # define GAN loss.
            self.criterionPR = ProsodyReconstructionLoss(train_config["loss"]["prosody"]["dist_loss_type"]).to(self.device, non_blocking=True)
            self.criterionRGR = ProsodyGuidedRegularizationLoss(sign_info, speaker_info, margin=train_config["loss"]["PGR"]["margin"]).to(self.device, non_blocking=True)
            self.criterionIR = IntonationRegularizationLoss().to(self.device, non_blocking=True)
            # learnable weight of Generative loss
            # self.weight_G = nn.Parameter(torch.tensor(0.), requires_grad=True)
            train_parameters = []
            train_layers = ["sign_processer", "s2s_mixier", "visual_project"]
            for name, param in self.netG_sign2audio.named_parameters():
                if any(layer_name in name for layer_name in train_layers):
                    train_parameters.append(param)
            train_parameters += list(self.netProsody_estimator.parameters())

            self.optimizer_G = torch.optim.AdamW(train_parameters,
                                                lr=train_config["optimizer"]["lr_G_s2a"],
                                                betas=train_config["optimizer"]["betas"],
                                                weight_decay=train_config["optimizer"]["weight_decay"])
            self.optimizer_D = torch.optim.AdamW(self.netD_audio.parameters(),
                                                lr=train_config["optimizer"]["lr_D_a"],
                                                betas=train_config["optimizer"]["betas"],
                                                weight_decay=train_config["optimizer"]["weight_decay"])
            # self.optimizer_w = torch.optim.AdamW([self.weight_G],
            #                                     lr=train_config["optimizer"]["lr_G_s2a"],
            #                                     betas=train_config["optimizer"]["betas"],
            #                                     weight_decay=train_config["optimizer"]["weight_decay"])
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_w)
            self.grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]

    def set_input(self, inputs):
        self.real_sign = inputs["sign"]
        self.real_audio = inputs["audio"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        batch_size = self.real_sign.text_tokens.shape[0]
        token_length = self.real_sign.token_length.to(self.device, non_blocking=True)
        max_src_len = token_length.max().to(self.device, non_blocking=True)
        text_tokens = self.real_sign.text_tokens[:, :max_src_len].to(self.device, non_blocking=True)

        speakers = random.choices(self.all_speakers, k=batch_size)
        speakers = torch.tensor(speakers, device=self.device).long()

        # with sign language TTS
        pred = self.netG_sign2audio(
            speakers,
            text_tokens,
            token_length,
            max_src_len,
            key_point=self.real_sign.visual_prefix.to(self.device, non_blocking=True)
        )
        # output without sign
        audio = pred.mels_wo_sign.masked_fill(
            pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.mels_wo_sign.shape[2]), 0.0).unsqueeze(1)
        audio_lens = pred.mel_lens.detach().cpu()
        prosody_predictions = torch.cat([
            pred.p_predictions_wo_sign.unsqueeze(1),
            pred.e_predictions_wo_sign.unsqueeze(1),
            torch.zeros_like(pred.weight_sign.detach(), device=self.device)
        ], dim=1)
        pred_prosody_label = self.netProsody_estimator(prosody_predictions)
        output_wo_sign = TrainOutput(
            audio, self.real_sign.token_length, audio_lens, pred.src_masks, pred.mel_masks,
            pred.p_predictions_wo_sign, pred.e_predictions_wo_sign, pred.log_d_predictions, pred.d_rounded,
            pred_prosody_label, None)

        # output with sign
        audio = pred.mels_w_sign.masked_fill(
            pred.mel_masks.unsqueeze(2).repeat(1, 1, pred.mels_w_sign.shape[2]), 0.0).unsqueeze(1)
        audio_lens = pred.mel_lens
        prosody_predictions = torch.cat([
            pred.p_predictions.unsqueeze(1),
            pred.e_predictions.unsqueeze(1),
            pred.weight_sign
        ], dim=1)
        pred_prosody_label = self.netProsody_estimator(prosody_predictions)
        output_w_sign = TrainOutput(
            audio, self.real_sign.token_length, audio_lens, pred.src_masks, pred.mel_masks,
            pred.p_predictions_w_sign, pred.e_predictions_w_sign, pred.log_d_predictions, pred.d_rounded,
            pred_prosody_label, pred.weight_sign)

        torch.cuda.empty_cache()
        return output_wo_sign, output_w_sign, speakers.detach().cpu().tolist()

    def calc_D(self, output: TrainOutput):
        fake, fake_lens = self.fake_audio_pool.query(
            output.mels.detach().unsqueeze(1).cpu(),
            output.mel_lens.detach().cpu()
            )
        fake = fake.to(self.device, non_blocking=True)

        real = self.real_audio.mels.unsqueeze(1).float().to(self.device, non_blocking=True)
        if real.dim() > 4:
            real = real.squeeze(1)
        real_lens = self.real_audio.mel_lens

        clip_length = min(min(fake_lens), min(real_lens)) + 8
        fake = random_clip_batch(fake, fake_lens, clip_length)
        real = random_clip_batch(real, real_lens, clip_length)

        pred_fake = self.netD_audio(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_real = self.netD_audio(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        torch.cuda.empty_cache()
        loss_log = {
            "GAN loss/D": loss_D.detach().cpu().item(),
            "output/pred_fake mean": pred_fake.detach().mean().cpu().item(),
            "output/pred_fake std": pred_fake.detach().std().cpu().item(),
            "output/pred_real mean": pred_real.detach().mean().cpu().item(),
            "output/pred_real std": pred_real.detach().std().cpu().item(),
        }
        return loss_D, loss_log

    def backward_D(self, output: TrainOutput):
        loss_D, loss_log = self.calc_D(output)
        loss_D.backward()
        torch.cuda.empty_cache()
        return loss_log

    def backward_G(self, output_wo_sign: TrainOutput, output_w_sign: TrainOutput, speakers):
        # GAN loss D_audio(G_sign2audio(sign))
        fake = output_w_sign.mels
        clip_length = output_w_sign.mel_lens.detach().min().cpu() + 8
        fake = random_clip_batch(fake, output_w_sign.mel_lens, clip_length)

        pred_fake = self.netD_audio(fake)
        loss_G_audio = self.criterionGAN(pred_fake, True)
        # loss_G_audio_weighted = loss_G_audio * F.softplus(self.weight_G.detach())

        # Prosody Reconstruction
        prosody_label = self.real_sign.prosody_label.to(self.device, non_blocking=True)
        loss_prosody, pr_info_w_sign = self.criterionPR(output_w_sign.sign_prosody_predictions, prosody_label)

        with torch.no_grad():
            _, pr_info_wo_sign = self.criterionPR(output_wo_sign.sign_prosody_predictions, prosody_label)

        # Prosody Guided Regularization
        loss_PGR, pgr_info = self.criterionRGR(output_w_sign, prosody_label, speakers)
        # Intonation Regularization
        loss_IR, ir_info = self.criterionIR(output_w_sign, output_wo_sign)
        # MoE weight regularization
        loss_gate = torch.abs(1 - output_w_sign.weight_sign.mean())
        # sum of loss around sign
        sign_loss = loss_prosody * self.weight_prosody + loss_PGR * self.weight_regdist_mean + loss_IR * self.weight_intotation + loss_gate * self.weight_gate
        # calculate loss for auto weight tuning
        # loss_weight_tune = torch.abs(loss_G_audio.detach() * F.softplus(self.weight_G) - loss_prosody.detach() * self.weight_prosody)
        # combined loss and calculate gradients
        loss_G = loss_G_audio + sign_loss # + loss_weight_tune
        loss_G.backward()

        loss_log = {
            # Loss log
            "prosody loss/v_loss": pr_info_w_sign.v_loss, "prosody loss without sign/v_loss": pr_info_wo_sign.v_loss,
            "prosody loss/a_loss": pr_info_w_sign.a_loss, "prosody loss without sign/a_loss": pr_info_wo_sign.a_loss,
            "prosody loss/total": pr_info_w_sign.total, "prosody loss without sign/total": pr_info_wo_sign.total,
            "Regularization/Energy mean": pgr_info.energy, "Regularization/Pitch mean": pgr_info.pitch,
            "Regularization/Energy intonation": ir_info.energy, "Regularization/Pitch intonation": ir_info.pitch,
            # "Regularization/Weight tune": loss_weight_tune.detach().cpu().item(),
            "GAN loss/G": loss_G_audio.detach().cpu().item(), # "GAN loss/G_weighted": loss_G_audio_weighted.detach().cpu().item(),
            "total": loss_G.detach().cpu().item(),
            # Output memo
            "output/weight_sign mean": output_w_sign.weight_sign.detach().mean().cpu().item(),
            "output/weight_sign std": output_w_sign.weight_sign.detach().std().cpu().item(),
            "output/pitch std (without sign)": output_wo_sign.p_predictions.detach().std(dim=1).mean().cpu().item(),
            "output/energy std (without sign)": output_wo_sign.e_predictions.detach().std(dim=1).mean().cpu().item(),
            "output/pitch std (with sign)": output_w_sign.p_predictions.detach().std(dim=1).mean().cpu().item(),
            "output/energy std (with sign)": output_w_sign.e_predictions.detach().std(dim=1).mean().cpu().item(),
        }

        torch.cuda.empty_cache()
        return loss_log

    def optimize_parameters(self, train_disc=True):
        loss_log = dict()
        self.set_train_mode()
        # forward
        output_wo_sign, output_w_sign, speakers = self.forward()

        # Generator's optimizing
        self.set_requires_grad([self.netD_audio], False)
        self.optimizer_G.zero_grad(set_to_none=True)
        loss_log_G = self.backward_G(output_wo_sign, output_w_sign, speakers)
        loss_log.update(loss_log_G)
        nn.utils.clip_grad_norm_(self.netG_sign2audio.parameters(), self.grad_clip_thresh)
        nn.utils.clip_grad_norm_(self.netProsody_estimator.parameters(), self.grad_clip_thresh)
        # nn.utils.clip_grad_norm_([self.weight_G], self.grad_clip_thresh)
        self.optimizer_G.step()
        # self.optimizer_w.step()
        # Discriminator's optimizing
        if random.random() < 0.5:
            self.real_audio = output_wo_sign
        if self.step == 0 and train_disc:
            self.set_requires_grad([self.netD_audio], True)
            self.optimizer_D.zero_grad(set_to_none=True)
            self.loss_log_D = self.backward_D(output_w_sign)
            nn.utils.clip_grad_norm_(self.netD_audio.parameters(), self.grad_clip_thresh)
            self.optimizer_D.step()
        else:
            with torch.no_grad():
                _, self.loss_log_D = self.calc_D(output_w_sign)
        self.step = (self.step + 1) % 2
        loss_log.update(self.loss_log_D)
        torch.cuda.empty_cache()
        return loss_log
