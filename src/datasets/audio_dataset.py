import os
import json
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset

from libs.util.tool import expand
from libs.preprocessing.speech.text import text_to_sequence
from libs.util.tool import pad_1D, pad_2D


AudioData = namedtuple("AudioData", "ids raw_texts speakers texts text_lens max_text_lens mels mel_lens max_mel_lens pitches energies durations")


class AudioDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, model_config, sort=False, drop_last=False,
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
            self.speaker_num = len(self.speaker_map)

        if model_config["speaker"]["all"] == "None":
            self.all_speakers = self.speaker_map.keys()
        else:
            self.all_speakers = set()
            with open(model_config["speaker"]["all"], "r") as f:
                for pid in f.readlines():
                    self.all_speakers.add(pid.rstrip())
            basename_, speaker_, text_, raw_text_ = [], [], [], []
            for i in range(len(self.text)):
                if self.speaker[i] not in self.all_speakers:
                    continue
                basename_.append(self.basename[i])
                speaker_.append(self.speaker[i])
                text_.append(self.text[i])
                raw_text_.append(self.raw_text[i])
            self.basename = basename_
            self.speaker = speaker_
            self.text = text_
            self.raw_text = raw_text_

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def get_speaker_info(self):
        info_pitch =    { self.speaker_map[s]: [] for s in self.all_speakers}
        info_energy =   { self.speaker_map[s]: [] for s in self.all_speakers}
        info_duration = { self.speaker_map[s]: [] for s in self.all_speakers}
        for idx in range(self.__len__()):
            basename = self.basename[idx]
            speaker = self.speaker[idx]
            speaker_id = self.speaker_map[speaker]
            pitch_path = os.path.join(
                self.preprocessed_path,
                "pitch",
                "{}-pitch-{}.npy".format(speaker, basename),
            )
            pitch = np.load(pitch_path)
            energy_path = os.path.join(
                self.preprocessed_path,
                "energy",
                "{}-energy-{}.npy".format(speaker, basename),
            )
            energy = np.load(energy_path)
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)

            info_pitch[speaker_id].append(expand(pitch, duration))
            info_energy[speaker_id].append(expand(energy, duration))
            info_duration[speaker_id].append(duration)

        info_pitch_mean = {}
        info_pitch_std = {}
        info_energy_mean = {}
        info_energy_std = {}
        info_duration_mean = {}
        info_duration_std = {}
        for speaker_id in info_pitch.keys():
            if len(info_pitch[speaker_id]) == 0:
                continue
            pitch = np.concatenate(info_pitch[speaker_id])
            energy = np.concatenate(info_energy[speaker_id])
            duration = np.concatenate(info_duration[speaker_id])
            info_pitch_mean[speaker_id] = pitch.mean()
            info_pitch_std[speaker_id] = pitch.std()
            info_energy_mean[speaker_id] = energy.mean()
            info_energy_std[speaker_id] = energy.std()
            info_duration_mean[speaker_id] = duration.mean()
            info_duration_std[speaker_id] = duration.std()
        info_pitch = {
            "mean": info_pitch_mean,
            "std": info_pitch_std,
        }
        info_energy = {
            "mean": info_energy_mean,
            "std": info_energy_std,
        }
        info_duration = {
            "mean": info_duration_mean,
            "std": info_duration_std,
        }
        return {
            "pitch": info_pitch,
            "energy": info_energy,
            "duration": info_duration }

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            lines = f.readlines()
            for line in lines:
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = torch.from_numpy(np.array([text.shape[0] for text in texts])).long()
        mel_lens = torch.from_numpy(np.array([mel.shape[0] for mel in mels])).long()

        speakers = torch.from_numpy(np.array(speakers)).long()
        texts = torch.from_numpy(pad_1D(texts))
        mels = torch.from_numpy(pad_2D(mels))
        pitches = torch.from_numpy(pad_1D(pitches))
        energies = torch.from_numpy(pad_1D(energies))
        durations = torch.from_numpy(pad_1D(durations))

        return AudioData(
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output
