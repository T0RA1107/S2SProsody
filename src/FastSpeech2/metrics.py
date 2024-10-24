import torch
from fastdtw import fastdtw


def calc_pitch_moment(pitch_prediction: torch.Tensor, pitch_target: torch.Tensor):
    mean_prediction = torch.mean(pitch_prediction, dim=1, keepdim=True)
    mean_target = torch.mean(pitch_target, dim=1, keepdim=True)
    sigma_prediction = torch.mean((pitch_prediction - mean_prediction) ** 2.0, dim=1)
    sigma_target = torch.mean((pitch_target - mean_target) ** 2.0, dim=1)
    gamma_prediction = torch.mean((pitch_prediction - mean_prediction) ** 3.0, dim=1)
    gamma_target = torch.mean((pitch_target - mean_target) ** 3.0, dim=1)
    kappa_prediction = torch.mean((pitch_prediction - mean_prediction) ** 4.0, dim=1)
    kappa_target = torch.mean((pitch_target - mean_target) ** 4.0, dim=1)
    return [
        (sigma_prediction, sigma_target),
        (gamma_prediction, gamma_target),
        (kappa_prediction, kappa_target)
    ]

def calc_pitch_dtw(pitch_prediction: torch.Tensor, pitch_target: torch.Tensor):
    dtw_sum = 0.
    for i in range(pitch_prediction.shape[0]):
        dtw_sum += fastdtw(pitch_prediction[i], pitch_target[i])[0]
    return dtw_sum / pitch_prediction.shape[0]
