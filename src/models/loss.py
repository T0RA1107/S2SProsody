from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


PRInfo = namedtuple("PRInfo", "v_loss a_loss total")
PGRInfo = namedtuple("PGRInfo", "energy pitch")


def sinkhorn_log(r, c, cost_matrix, lambd=1.0, num_iters=100, eps=1e-8):
    K = -cost_matrix * lambd
    batch_size = r.shape[0]
    device = r.device
    K = K.unsqueeze(0).repeat(batch_size, 1, 1)
    logu = torch.zeros_like(r, device=device)
    logv = torch.zeros_like(c, device=device)
    logr = torch.log(r + eps)
    logc = torch.log(c + eps)
    for _ in range(num_iters):
        logu = logr - torch.logsumexp(K + logv.unsqueeze(1), axis=2)
        logv = logc - torch.logsumexp(K + logu.unsqueeze(2), axis=1)
    P = torch.exp(logu.unsqueeze(2) + K + logv.unsqueeze(1))
    d = torch.sum(P * cost_matrix, dim=(1, 2))
    return d


class EarthMoversDistanceLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, label):
        dim = pred.shape[-1]
        pred = F.softmax(pred, dim=-1)
        bins = torch.linspace(0, 1, dim + 1)
        x = (bins[1:] + bins[:-1]) / 2
        cost_matrix = torch.abs(x.reshape(1, -1).repeat(dim, 1) - x.reshape(-1, 1).repeat(1, dim))
        ret = sinkhorn_log(pred, label, cost_matrix.to(pred.device))
        if self.reduction == "mean":
            return ret.mean()
        elif self.reduction == "sum":
            return ret.sum()
        else:
            return ret


class ProsodyReconstructionLoss(nn.Module):
    def __init__(self, dist_loss_type="CELoss"):
        super().__init__()
        if dist_loss_type == "CELoss":
            self.distribution_loss = nn.CrossEntropyLoss(reduction="none")
        elif dist_loss_type == "EMDLoss":
            self.distribution_loss = EarthMoversDistanceLoss(reduction="none")
        else:
            raise NotImplementedError()

    def forward(self, pred_prosody_label, prosody_label):
        loss_log = {}
        batch_size = pred_prosody_label.shape[0]
        pred_prosody_label = rearrange(pred_prosody_label, "b c d -> (b c) d")
        prosody_label = rearrange(prosody_label, "b c d -> (b c) d")
        loss_prosody = self.distribution_loss(pred_prosody_label, prosody_label)
        loss_prosody = rearrange(loss_prosody, "(b c) -> b c", c=4).sum(0) / batch_size

        loss_prosody_total = sum(loss_prosody) / 4.
        loss_log["prosody loss/total"] = loss_prosody_total.detach().cpu()
        return loss_prosody_total, PRInfo(sum(loss_prosody[:2]).detach().cpu() / 2., sum(loss_prosody[2:4]).detach().cpu() / 2., loss_prosody_total.detach().cpu())


class ProsodyGuidedRegularizationLoss(nn.Module):
    def __init__(self, sign_info, speaker_info, margin=0.):
        super().__init__()
        self.sign_info = sign_info
        self.speaker_info = speaker_info
        self.margin = margin

    def forward(self, audio_predictions, sign_prosody_label, speakers):
        device = audio_predictions.e_predictions.device
        # calculate speaker-wise standardization
        energy_mean_info = torch.tensor([self.speaker_info["energy"]["mean"][i] for i in speakers], device=device).float()
        energy_std_info  = torch.tensor([self.speaker_info["energy"]["std"][i] for i in speakers], device=device).float()
        pitch_mean_info  = torch.tensor([self.speaker_info["pitch"]["mean"][i] for i in speakers], device=device).float()
        pitch_std_info   = torch.tensor([self.speaker_info["pitch"]["std"][i] for i in speakers], device=device).float()

        L = audio_predictions.e_predictions.shape[1]
        token_length = (~audio_predictions.src_masks).sum(dim=1)
        energy_mean_info = energy_mean_info.unsqueeze(1).repeat(1, L)
        energy_std_info = energy_std_info.unsqueeze(1).repeat(1, L)
        norm_energy = (audio_predictions.e_predictions - energy_mean_info) / (energy_std_info + 1e-5)
        norm_energy = norm_energy.masked_fill(audio_predictions.src_masks, 0.0)
        energy_mean = norm_energy.sum(dim=1) / token_length
        # energy_var = (norm_energy ** 2).sum(dim=1) / token_length - energy_mean ** 2

        pitch_mean_info = pitch_mean_info.unsqueeze(1).repeat(1, L)
        pitch_std_info = pitch_std_info.unsqueeze(1).repeat(1, L)
        norm_pitch = (audio_predictions.p_predictions - pitch_mean_info) / (pitch_std_info + 1e-5)
        norm_pitch = norm_pitch.masked_fill(audio_predictions.src_masks, 0.0)
        pitch_mean = norm_pitch.sum(dim=1) / token_length
        # pitch_var = (norm_pitch ** 2).sum(dim=1) / token_length - pitch_mean ** 2

        bs, _, bins = sign_prosody_label.shape
        x = torch.arange(0, bins) / bins + 1 / (2 * bins)
        x = x.unsqueeze(0).repeat(bs, 1).to(device, non_blocking=True)
        x_velocity_hand = (x - self.sign_info["mean"][0]) / (self.sign_info["std"][0] + 1e-5)
        x_velocity_face = (x - self.sign_info["mean"][1]) / (self.sign_info["std"][1] + 1e-5)
        velocity_hand_mean = (sign_prosody_label[:, 0] * x_velocity_hand).sum(dim=1)
        # velocity_hand_var = (sign_prosody_label[:, 0] * (x_velocity_hand - velocity_hand_mean.unsqueeze(1)) ** 2).sum(dim=1)
        velocity_face_mean = (sign_prosody_label[:, 1] * x_velocity_face).sum(dim=1)
        # velocity_face_var = (sign_prosody_label[:, 1] * (x_velocity_face - velocity_face_mean.unsqueeze(1)) ** 2).sum(dim=1)

        loss_reg_energy_mean = F.relu(torch.abs(energy_mean - velocity_hand_mean) - self.margin).mean()
        loss_reg_pitch_mean  = F.relu(torch.abs(pitch_mean  - velocity_face_mean) - self.margin).mean()
        loss_total = loss_reg_energy_mean + loss_reg_pitch_mean
        return loss_total, PGRInfo(loss_reg_energy_mean.detach().cpu(), loss_reg_pitch_mean.detach().cpu())