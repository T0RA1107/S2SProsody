import re
import json
import os
from string import punctuation
import random
from collections import namedtuple
from multiprocessing import Pool

import math
import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from g2p_en import G2p

from FastSpeech2.text import text_to_sequence
from FastSpeech2.utils.tools import pad_1D, pad_2D


AudioData = namedtuple("AudioData", "ids raw_texts speakers texts text_lens max_text_lens mels mel_lens max_mel_lens pitches energies durations")
SignTrainData = namedtuple("SignTrainData", "raw_texts text_tokens mask visual_prefix token_length visual_length prosody_label video_names")
SignTestData = namedtuple("SignTestData", "raw_texts visual_prefix index visual_length")


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


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

        self.target_speaker = model_config["speaker"]["target"]
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


class SignDataset(Dataset):
    def __init__(self, args, preprocess_config, train_config,
                 phase="train", split="train", partial_list_path=None, sort=False, drop_last=False):
        self.sort = sort
        self.drop_last = drop_last
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.phase = phase
        self.split = split
        self.partial = partial_list_path is not None
        self.prosody_dist = train_config["loss"]["prosody"]["dist"]
        if self.prosody_dist:
            self.prosody_bins = train_config["loss"]["prosody"]["bins"]

        # path to openasl
        self.feat_path = preprocess_config["preprocessing_sign"]["feat_path"]
        self.label_path = preprocess_config["preprocessing_sign"]["label_path"]
        assert self.feat_path is not None and self.feat_path is not None

        self.local_rank = args.local_rank
        self.eos_token = preprocess_config["preprocessing_sign"]["eos_token"]
        # information about input clips (features)
        self.visual_token_num = preprocess_config["preprocessing_sign"]["clip_length"]
        self.visual_token_dim = preprocess_config["preprocessing_sign"]["prefix_dim"]


        # label_path = os.path.join(f"/mnt/workspace/openasl-pre/openasl-v1.0.tsv")
        assert self.label_path is not None, "Specify --label_path"
        data_frame = pd.read_csv(self.label_path, sep="\t", low_memory=False)

        if self.partial:
            partial_mp4_list = open(partial_list_path).readlines()
            partial_id_list = [s.split("/")[-1].rstrip(".mp4\n") for s in partial_mp4_list]
            self.partial_list = partial_id_list

            partial_mask = data_frame.vid == ""
            for vid in partial_id_list:
                partial_mask |= data_frame.vid == vid
        else:
            # select by split ["train", "valid"]
            data_frame = data_frame.loc[data_frame["split"].str.contains(split)]

        def filter_missing_or_short(row):
            full_path = os.path.join(self.feat_path, self.split, f"{row['vid']}.pkl")
            ok = os.path.exists(full_path) and os.path.getsize(full_path) > 0
            if ok and self.phase == "train":
                with open(full_path, "rb") as file:
                    pose_keypoints = pickle.load(file)
                    ok = ok and (pose_keypoints.shape[0] >= 30)
            return ok

        is_missing_or_short = data_frame.apply(filter_missing_or_short, axis=1)
        df_filtered = data_frame[is_missing_or_short]
        if self.local_rank == 0:
            base_name = os.path.basename(self.label_path)
            df_filtered.to_csv(self.label_path.replace(base_name, f"filtered_{self.split}.tsv"), sep="\t")
        if self.local_rank == 0:
            print(f"Split:{split}\nBefore filtering: {len(data_frame)}\n After filtering: {len(df_filtered)}")
        # translation labels and sample names (split agnostic)
        self.translation = df_filtered["raw-text"].to_list()
        self.video_names = df_filtered["vid"].to_list()
        self.yids = df_filtered["yid"].to_list()
        self.vid2idx = {
            vid: i for i, vid in enumerate(self.video_names)
        }

        token_id_lens = [0 for _ in range(len(self.translation))]
        self.translation_token_ids = [np.array([]) for _ in range(len(self.translation))]
        if os.path.exists(preprocess_config["preprocessing_sign"]["translation_token_ids"]):
            for t in open(preprocess_config["preprocessing_sign"]["translation_token_ids"]).readlines():
                vid, trans_ids_str = t.rstrip("\n").split("|")
                trans_ids = list(map(int, trans_ids_str.split(" ")))
                if vid in self.vid2idx:
                    self.translation_token_ids[self.vid2idx[vid]] = np.array(trans_ids)
                    token_id_lens[self.vid2idx[vid]] = len(trans_ids)
        if not all(token_id_lens):
            raise ValueError("Prepare appropriate translation_token_ids file.")

        ### Filtering too long translation_token_ids
        if not self.partial and self.phase == "train":
            trans_id_arg = np.argsort(token_id_lens)
            token_id_lens = np.array(token_id_lens)[trans_id_arg]
            too_long_idx = -1
            while token_id_lens[too_long_idx] > 250:
                too_long_idx -= 1
            self.translation_token_ids = [
                self.translation_token_ids[trans_id_arg[i]] for i in range(len(trans_id_arg))][:too_long_idx]
            self.translation = [
                self.translation[trans_id_arg[i]] for i in range(len(trans_id_arg))][:too_long_idx]
            self.video_names = [
                self.video_names[trans_id_arg[i]] for i in range(len(trans_id_arg))][:too_long_idx]
            self.yids = [
                self.yids[trans_id_arg[i]] for i in range(len(trans_id_arg))][:too_long_idx]
            self.vid2idx = {
                vid: i for i, vid in enumerate(self.video_names)
            }
        if self.partial:
            self.partial_idx = []
            for vid in self.partial_list:
                self.partial_idx.append(self.vid2idx[vid])

        ### start generate video-wise normalization stats ###
        yid2idxs = { yid: [] for yid in self.yids }
        for i, yid in enumerate(self.yids):
            yid2idxs[yid].append(i)

        # x_min, x_max, y_min, y_max
        self.yid2range = {}
        for yid, idxs in yid2idxs.items():
            key_points = []
            for i in idxs:
                key_points.append(self.read_keypoints(i))
            key_points = np.concatenate(key_points, axis=0)
            self.yid2range[yid] = [
                key_points[:, :, 0].min(),
                key_points[:, :, 0].max(),
                key_points[:, :, 1].min(),
                key_points[:, :, 1].max()
            ]
        ### end generate video-wise normalization stats ###

        assert len(self.translation_token_ids) == len(
            self.video_names), f"Text ids count:{len(self.translation_token_ids)}\tVid count:{len(self.video_names)}"
        all_len = np.array([len(tk)
                               for tk in self.translation_token_ids])
        if not self.partial and self.phase == "train":
            self.max_seq_len = min(
                int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        else:
            self.max_seq_len = int(all_len.max())
        if self.local_rank == 0:
            print(f"Max sequence length:{self.max_seq_len}")

    def __len__(self):
        if self.partial:
            return len(self.partial_idx)
        return len(self.vid2idx)

    def get_sign_prosody_info(self):
        prosody_labels = []
        for i in range(len(self.vid2idx)):
            sign = self.__getitem__(i)
            prosody_labels.append(sign.prosody_label[None,])
        prosody_labels = np.concatenate(prosody_labels)
        prosody_dist = prosody_labels.sum(axis=0) / prosody_labels.sum()

        n, bins = prosody_dist.shape
        x = np.arange(0, bins) / bins + 1 / (2 * bins)
        x = x[None,].repeat(n, axis=0)
        mean = (x * prosody_dist).sum(axis=1)
        var  = ((x - mean[:, None]) ** 2 * prosody_dist).sum(axis=1)
        std = var ** 0.5
        return { "mean": mean, "std": std }

    def read_keypoints(self, index: int):
        vid_name = self.video_names[index]
        file_path = os.path.join(self.feat_path, self.split, f"{vid_name}.pkl")
        with open(file_path, "rb") as f:
            pose_keypoints = pickle.load(f)
        return pose_keypoints

    def read_video_wise_normal_keypoints(self, index: int):
        pose_keypoints = self.read_keypoints(index)
        yid = self.yids[index]
        video_range = self.yid2range[yid]
        pose_keypoints[:, :, 0] = (pose_keypoints[:, :, 0] - video_range[0]) / (video_range[1] - video_range[0] + 1e-5)
        pose_keypoints[:, :, 1] = (pose_keypoints[:, :, 1] - video_range[2]) / (video_range[3] - video_range[2] + 1e-5)
        pose_keypoints[:, :, :2] = pose_keypoints[:, :, :2] * 2 - 1
        return pose_keypoints

    def make_prosody_label(self, pose_keypoints):
        represnt_hands = pose_keypoints[:, [91, 112], :]
        represent_face = pose_keypoints[:, [50, 85, 42, 47], :]

        def norm_pose(pose):
            return (pose[:, :, 0] ** 2 + pose[:, :, 1] ** 2) ** 0.5

        velocity_hands = np.diff(represnt_hands[:, :, :2], axis=0)
        velocity_face = np.diff(represent_face[:, :, :2], axis=0)

        accel_hands = np.diff(velocity_hands, axis=0)
        accel_face = np.diff(velocity_face, axis=0)

        velocity_cated = np.concatenate((
            norm_pose(velocity_hands).sum(axis=1, keepdims=True),
            norm_pose(velocity_face).sum(axis=1, keepdims=True)),
        axis=1)

        accel_cated = np.concatenate((
            norm_pose(accel_hands).sum(axis=1, keepdims=True),
            norm_pose(accel_face).sum(axis=1, keepdims=True)),
        axis=1)

        if not self.prosody_dist:
            velocity_max = np.max(velocity_cated, axis=0)
            accel_max = np.max(accel_cated, axis=0)
            prosody_label = np.concatenate(
                (velocity_max, accel_max), axis=0)
        else:
            v_max = 2 * 2 ** 0.5
            a_max = 4 * 2 ** 0.5
            bins = self.prosody_bins
            eps = 1e-4
            v_edges = np.linspace(np.log(eps), np.log(v_max), bins + 1)
            a_edges = np.linspace(np.log(eps), np.log(a_max), bins + 1)
            log_v = np.log(velocity_cated + eps)
            log_a = np.log(accel_cated + eps)
            hist = np.concatenate((
                np.histogram(log_v[:, 0], bins=v_edges)[0][None,],
                np.histogram(log_v[:, 1], bins=v_edges)[0][None,],
                np.histogram(log_a[:, 0], bins=a_edges)[0][None,],
                np.histogram(log_a[:, 1], bins=a_edges)[0][None,],), axis=0)
            prosody_label = hist.astype(np.float32) / hist.sum(axis=1, keepdims=True)
            assert prosody_label.shape == (4, bins)

        return prosody_label

    def read_pose_files(self, index: int):
        # MMPose 76
        body_sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        face_sample_indices = [71, 77, 85, 89] + \
                              [40, 42, 44, 45, 47, 49] + \
                              [59, 60, 61, 62, 63, 64] + [65, 66, 67, 68, 69, 70] + \
                              [50]

        # read files
        pose_keypoints = self.read_video_wise_normal_keypoints(index)

        # 23(17+6) 11 selected
        body_pose = pose_keypoints[:, body_sample_indices, :]
        hand_right = pose_keypoints[:, 91:112, :]  # 21 Keypoints
        hand_left = pose_keypoints[:, 112:, :]  # 21 Keypoints
        face = pose_keypoints[:, face_sample_indices, :]  # 23 Keypoints

        prosody_label = self.make_prosody_label(pose_keypoints)

        pose_tuple = (body_pose, hand_left, hand_right, face)
        pose_cated = np.concatenate(
            pose_tuple, axis=1)  # [F, 11+21+21+23=76, 3]

        # scale to [-1, 1]
        # normalization, same as pre-training, might not align with actual output shape which assumed to be [288, 384]
        pose_cated[:, :, 0:2] = 2.0 * ((pose_cated[:, :, 0:2] / 256.0) - 0.5)

        # pad pose
        T, V, C = pose_cated.shape
        if T < self.visual_token_num:
            diff = self.visual_token_num - T
            pose_output = np.concatenate(
                (pose_cated, np.zeros((diff, V, C))), axis=0)
        elif T > self.visual_token_num:
            if self.phase == "train":
                diff = T - self.visual_token_num
                offset = np.random.randint(0, diff)
                pose_output = pose_cated[offset: offset +
                                     self.visual_token_num, :, :]
            elif self.phase == "test":
                offset = 0
                pose_output = pose_cated[offset: offset +
                                     self.visual_token_num, :, :]
        else:
            pose_output = pose_cated

        pose_length = T if T <= self.visual_token_num else self.visual_token_num
        assert pose_output.shape[0] >= self.visual_token_num, "{} {}".format(pose_output.shape[0], self.visual_token_num)

        return pose_output, pose_length, prosody_label

    def rand_view_transform(self, X, agx, agy, s):
        if X.shape[-1] == 2:
            padding = np.zeros((X.shape[0], X.shape[1], 1))
            X = np.concatenate((X, padding), axis=2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,              0,             0],
                         [0,  math.cos(agx), math.sin(agx)],
                         [0, -math.sin(agx), math.cos(agx)]])

        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)],
                         [0, 1,              0],
                         [math.sin(agy), 0,  math.cos(agy)]])

        Ss = np.asarray([[s, 0, 0],
                         [0, s, 0],
                         [0, 0, s]])

        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def normalize_joints(self, value):
        T, V, C = value.shape
        # scale to [-1, 1]
        scalerValue = np.reshape(value, (-1, C))
        scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / \
            ((np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0)) + 1e-5)

        scalerValue = scalerValue * 2 - 1
        scalerValue = np.reshape(scalerValue, (-1, V, C))

        return scalerValue

    def pad_token_ids(self, index: int, pad_const=0):
        tokens = self.translation_token_ids[index]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = np.concatenate((tokens, np.zeros(padding, dtype=np.int64)))
            self.translation_token_ids[index] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.translation_token_ids[index] = tokens
        mask = tokens > 0  # mask is zero where we out of sequence
        tokens_length = mask.sum()
        tokens[~mask] = pad_const
        mask = mask.astype(np.float32)

        return tokens, mask, tokens_length

    def __getitem__(self, index: int):
        if self.partial:
            index = self.partial_idx[index]
        if self.phase == "train":
            raw_texts = self.translation[index]
            text_tokens, mask, token_length = self.pad_token_ids(
                index)  # [max_seq_len]
            visual_prefix, visual_length, prosody_label = self.read_pose_files(index)
            video_name = self.video_names[index]
            agx = np.random.randint(-60, 60)
            agy = np.random.randint(-60, 60)
            s = np.random.uniform(0.5, 1.5)
            # augmentation
            # visual_prefix[:, :, :2] = self.rand_view_transform(
            #     visual_prefix[:, :, :2], agx, agy, s)[:, :, :2]
            # visual_prefix[:, :, :2] = self.normalize_joints(
            #     visual_prefix[:, :, :2])
            assert visual_prefix.shape[0] >= self.visual_token_num, "{} {}".format(visual_prefix.shape[0], self.visual_token_num)
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            # vn_idxs, vn_len = self.get_vn(index)
            return SignTrainData(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_name)  # , vn_idxs, vn_len
        elif self.phase == "test":
            raw_texts = self.translation[index]
            text_tokens, mask, token_length = self.pad_token_ids(
                index)  # [max_seq_len]
            visual_prefix, visual_length, _ = self.read_pose_files(index)
            video_name = self.video_names[index]
            # visual_prefix[:, :, :2] = self.normalize_joints(
            #     visual_prefix[:, :, :2])
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            return SignTrainData(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, _, video_name)

    def reprocess(self, raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_names, idxs):
        raw_texts = [raw_texts[idx] for idx in idxs]
        text_tokens = torch.from_numpy(np.array([text_tokens[idx] for idx in idxs])).long()
        mask = torch.from_numpy(np.array([mask[idx] for idx in idxs])).float()
        visual_prefix = torch.from_numpy(np.array([visual_prefix[idx] for idx in idxs])).float()
        token_length = torch.from_numpy(np.array([token_length[idx] for idx in idxs]))
        visual_length = torch.from_numpy(np.array([visual_length[idx] for idx in idxs]))
        prosody_label = torch.from_numpy(np.array([prosody_label[idx] for idx in idxs])).float()
        video_names = [video_names[idx] for idx in idxs]


        return SignTrainData(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_names)

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d[1].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        raw_texts = [d[0] for d in data]
        text_tokens = np.stack([d[1] for d in data], axis=0)
        mask = np.stack([d[2] for d in data], axis=0)
        visual_prefix = np.stack([d[3] for d in data], axis=0)
        token_length = np.stack([d[4] for d in data], axis=0)
        visual_length = [d[5] for d in data]
        prosody_label = np.stack([d[6] for d in data], axis=0)
        video_names = [d[7] for d in data]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(
                raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_names, idx))

        return output


class UnpairedAudioSignDataset(Dataset):

    def __init__(self, filename, preprocess_config, train_config, model_config,
                 args, phase="train", split="train", partial_list_path=None):
        self.audio_dataset = AudioDataset(filename, preprocess_config, train_config, model_config)
        self.sign_dataset = SignDataset(args, preprocess_config, train_config, phase, split, partial_list_path)

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
