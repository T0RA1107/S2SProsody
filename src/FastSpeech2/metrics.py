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

def normalize_audio(pitch: torch.Tensor):
    """
    pitch: [B, T]
    """
    mean = torch.mean(pitch, dim=1, keepdim=True)
    sigma = torch.mean((pitch - mean) ** 2, dim=1, keepdim=True) ** 0.5
    return (pitch - mean) / sigma

def pitch_based_align(pitch1: torch.Tensor, pitch2: torch.Tensor):
    assert pitch1.shape[0] == pitch2.shape[0], "Batch size should be same"
    pitch1 = normalize_audio(pitch1)
    pitch2 = normalize_audio(pitch2)
    paths = []
    for i in range(pitch1.shape[0]):
        _, path = fastdtw(pitch1[i], pitch2[i])
        paths.append(path)
    return paths

def pitch_based_pearson_correlation(pitch1, pitch2, target1, target2):
    assert pitch1.shape[0] == pitch2.shape[0], "Batch size should be same"
    assert target1.shape[0] == target2.shape[0], "Batch size should be same"
    paths = pitch_based_align(pitch1, pitch2)
    pc = []
    for i in range(pitch1.shape[0]):
        n = len(paths[i])
        x_mean = sum(target1[x] for x, _ in paths[i]) / n
        y_mean = sum(target2[y] for _, y in paths[i]) / n
        sx = 0
        sy = 0
        sxy = 0
        for x, y in paths[i]:
            sx += (x - x_mean) ** 2
            sy += (y - y_mean) ** 2
            sxy += (x - x_mean) * (y - y_mean)
        pc.append(sxy / (sx * sy) ** 0.5)
    return pc
