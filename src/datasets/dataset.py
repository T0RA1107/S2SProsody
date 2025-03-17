import random

from torch.utils.data import Dataset

from .audio_dataset import AudioDataset
from .sign_dataset import SignDataset


class UnpairedAudioSignDataset(Dataset):

    def __init__(self, filename, preprocess_config, train_config, model_config,
                 local_rank, phase="train", split="train", partial_list_path=None):
        self.audio_dataset = AudioDataset(filename, preprocess_config, train_config, model_config)
        self.sign_dataset = SignDataset(preprocess_config, train_config, local_rank, phase, split, partial_list_path)

        self.audio_size = len(self.audio_dataset)
        self.sign_size = len(self.sign_dataset)

        self.speaker_num = self.audio_dataset.speaker_num

        self.serial_batches = train_config["dataset"]["serial_batches"]

    def __len__(self):
        return len(self.sign_dataset)

    def __getitem__(self, index):
        idx_sign = index % self.sign_size
        if self.serial_batches:
            idx_audio = index % self.audio_dataset
        else:
            idx_audio = random.randint(0, self.audio_size - 1)
        return {
            "audio": self.audio_dataset.__getitem__(idx_audio),
            "sign": self.sign_dataset.__getitem__(idx_sign)
        }

    def collate_fn(self, data):
        audio = [d["audio"] for d in data]
        sign = [d["sign"] for d in data]
        audio_collated = self.audio_dataset.collate_fn(audio)
        sign_collated = self.sign_dataset.collate_fn(sign)

        output = list()
        for a, s in zip(audio_collated, sign_collated):
            output.append({
                "audio": a,
                "sign": s
            })
        return output
