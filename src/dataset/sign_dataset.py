from logging import getLogger
from pathlib import Path
from typing import Tuple, Dict, List, Any
from collections import namedtuple

import math
import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset


logger = getLogger(__name__)

SignData = namedtuple("SignData", "raw_texts text_tokens mask visual_prefix token_length visual_length prosody_label video_names")


class SignDataset(Dataset):
    def __init__(self, preprocess_config: Dict[str, Any], train_config: Dict[str, Any], local_rank: int,
                 phase: str="train", split: str="train", partial_list_path: str=None, sort: bool=False, drop_last: bool=False):
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

        self.local_rank = local_rank
        self.eos_token = preprocess_config["preprocessing_sign"]["eos_token"]
        # information about input clips (features)
        self.visual_token_num = preprocess_config["preprocessing_sign"]["clip_length"]
        self.visual_token_dim = preprocess_config["preprocessing_sign"]["prefix_dim"]


        assert self.label_path is not None, "Specify --label_path"
        data_frame = pd.read_csv(self.label_path, sep="\t", low_memory=False)

        if self.partial:
            partial_mp4_list = open(partial_list_path).readlines()
            partial_gender_list = ["male" if s[0] == "M" else "female" for s in partial_mp4_list]
            partial_id_list = [s.split("|")[1].split("/")[-1].rstrip(".mp4\n") for s in partial_mp4_list]
            self.partial_list = partial_id_list
            self.partial_vid2gender = { vid: g for vid, g in zip(partial_id_list, partial_gender_list) }

            partial_mask = data_frame.vid == ""
            for vid in partial_id_list:
                partial_mask |= data_frame.vid == vid
        else:
            data_frame = data_frame.loc[data_frame["split"].str.contains(split)]

        def filter_missing_or_short(row):
            full_path = Path(self.feat_path, self.split, f"{row['vid']}.pkl")
            ok = full_path.exists() and full_path.stat().st_size > 0
            if ok and self.phase == "train":
                with full_path.open("rb") as file:
                    pose_keypoints = pickle.load(file)
                    ok = ok and (pose_keypoints.shape[0] >= 30)
            return ok

        is_missing_or_short = data_frame.apply(filter_missing_or_short, axis=1)
        df_filtered = data_frame[is_missing_or_short]
        if self.local_rank == 0:
            base_name = Path(self.label_path).name
            df_filtered.to_csv(self.label_path.replace(base_name, f"filtered_{self.split}.tsv"), sep="\t")
        if self.local_rank == 0:
            logger.info(f"Split:{split}\nBefore filtering: {len(data_frame)}\nAfter filtering : {len(df_filtered)}")
        # translation labels and sample names (split agnostic)
        self.translation = df_filtered["raw-text"].to_list()
        self.video_names = df_filtered["vid"].to_list()
        self.yids = df_filtered["yid"].to_list()
        self.vid2idx = {
            vid: i for i, vid in enumerate(self.video_names)
        }

        token_id_lens = [0 for _ in range(len(self.translation))]
        self.translation_token_ids = [np.array([]) for _ in range(len(self.translation))]
        trans_id_path = Path(preprocess_config["preprocessing_sign"]["translation_token_ids"])
        if trans_id_path.exists():
            for t in trans_id_path.open().readlines():
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
        all_len = np.array([len(tk) for tk in self.translation_token_ids])
        if not self.partial and self.phase == "train":
            self.max_seq_len = min(
                int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        else:
            self.max_seq_len = int(all_len.max())
        if self.local_rank == 0:
            logger.info(f"Max sequence length:{self.max_seq_len}")

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
        prosody_dist = prosody_labels.sum(axis=0) / len(self.vid2idx)
        assert np.all(np.abs(prosody_dist.sum(axis=1) - 1.) <= 1e-8)

        n, bins = prosody_dist.shape
        x = np.arange(0, bins) / bins + 1 / (2 * bins)
        x = x[None,].repeat(n, axis=0)
        mean = (x * prosody_dist).sum(axis=1)
        var  = ((x - mean[:, None]) ** 2 * prosody_dist).sum(axis=1)
        std = var ** 0.5
        return { "mean": mean, "std": std }

    def read_keypoints(self, index: int) -> npt.NDArray[np.float32]:
        vid_name = self.video_names[index]
        file_path = Path(self.feat_path, self.split, f"{vid_name}.pkl")
        with file_path.open("rb") as f:
            pose_keypoints = pickle.load(f)
        return pose_keypoints

    def read_video_wise_normal_keypoints(self, index: int) -> npt.NDArray[np.float32]:
        pose_keypoints = self.read_keypoints(index)
        yid = self.yids[index]
        video_range = self.yid2range[yid]
        pose_keypoints[:, :, 0] = (pose_keypoints[:, :, 0] - video_range[0]) / (video_range[1] - video_range[0] + 1e-5)
        pose_keypoints[:, :, 1] = (pose_keypoints[:, :, 1] - video_range[2]) / (video_range[3] - video_range[2] + 1e-5)
        pose_keypoints[:, :, :2] = pose_keypoints[:, :, :2] * 2 - 1
        return pose_keypoints

    def make_prosody_label(self, pose_keypoints: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        represnt_hands = pose_keypoints[:, [91, 112], :]
        represent_face = pose_keypoints[:, [50, 85, 42, 47], :]

        def norm_pose(pose: npt.NDArray[np.float32]):
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

    def read_pose_files(self, index: int) -> Tuple[npt.NDArray[np.float32], int, npt.NDArray[np.float32]]:
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

    def __getitem__(self, index: int) -> SignData:
        if self.partial:
            index = self.partial_idx[index]
        if self.phase == "train":
            raw_texts = self.translation[index]
            text_tokens, mask, token_length = self.pad_token_ids(
                index)  # [max_seq_len]
            visual_prefix, visual_length, prosody_label = self.read_pose_files(index)
            video_name = self.video_names[index]
            assert visual_prefix.shape[0] >= self.visual_token_num, "{} {}".format(visual_prefix.shape[0], self.visual_token_num)
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            return SignData(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_name)
        elif self.phase == "test":
            raw_texts = self.translation[index]
            text_tokens, mask, token_length = self.pad_token_ids(
                index)  # [max_seq_len]
            visual_prefix, visual_length, _ = self.read_pose_files(index)
            video_name = self.video_names[index]
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            return SignData(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, _, video_name)

    def reprocess(self, raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_names, idxs) -> SignData:
        raw_texts = [raw_texts[idx] for idx in idxs]
        text_tokens = torch.from_numpy(np.array([text_tokens[idx] for idx in idxs])).long()
        mask = torch.from_numpy(np.array([mask[idx] for idx in idxs])).float()
        visual_prefix = torch.from_numpy(np.array([visual_prefix[idx] for idx in idxs])).float()
        token_length = torch.from_numpy(np.array([token_length[idx] for idx in idxs]))
        visual_length = torch.from_numpy(np.array([visual_length[idx] for idx in idxs]))
        prosody_label = torch.from_numpy(np.array([prosody_label[idx] for idx in idxs])).float()
        video_names = [video_names[idx] for idx in idxs]

        return SignData(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, prosody_label, video_names)

    def collate_fn(self, data) -> List[SignData]:
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
