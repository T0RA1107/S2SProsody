import re
import json
import os
from string import punctuation
import random
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


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    return text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )


def write_txt(pack):
    i, (vid, trans, preprocess_config) = pack
    trans_ids = preprocess_english(trans, preprocess_config)
    trans_ids_txt = "{}|{}\n".format(vid, " ".join(map(str, trans_ids)))
    return i, torch.tensor(trans_ids), trans_ids_txt


class AudioDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
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
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

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
            l = len(lines)
            for line in tqdm(lines, desc="process meta data of Audio", total=l):
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

        return (
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

        # path to openasl
        self.feat_path = preprocess_config["preprocessing_sign"]["feat_path"]
        self.label_path = preprocess_config["preprocessing_sign"]["label_path"]
        assert self.feat_path is not None and self.feat_path is not None

        self.local_rank = 0
        self.eos_token = preprocess_config["preprocessing_sign"]["eos_token"]
        # information about input clips (features)
        self.visual_token_num = preprocess_config["preprocessing_sign"]["clip_length"]
        self.visual_token_dim = preprocess_config["preprocessing_sign"]["prefix_dim"]


        # label_path = os.path.join(f"/mnt/workspace/openasl-pre/openasl-v1.0.tsv")
        assert self.label_path is not None, "Specify --label_path"
        data_frame = pd.read_csv(self.label_path, sep="\t")
        print(len(data_frame))

        if partial_list_path is not None:
            partial_mp4_list = open(partial_list_path).readlines()
            partial_id_list = [s.split("/")[-1].rstrip(".mp4\n") for s in partial_mp4_list]

            partial_mask = data_frame.vid == ""
            for vid in partial_id_list:
                partial_mask |= data_frame.vid == vid
            data_frame = data_frame[partial_mask]
        else:
            # select by split ["train", "valid"]
            data_frame = data_frame.loc[data_frame["split"].str.contains(split)]

        def filter_missing(row):
            full_path = os.path.join(self.feat_path, f"{row['vid']}.pkl")
            return os.path.exists(full_path) and os.path.getsize(full_path) > 0

        is_missing = data_frame.apply(filter_missing, axis=1)
        df_filtered = data_frame[is_missing]
        if self.local_rank == 0:
            print(f"Split:{split}\nBefore filtering: {len(data_frame)}\n After filtering: {len(df_filtered)}")
        # translation labels and sample names (split agnostic)
        self.translation = df_filtered["raw-text"].to_list()
        self.video_names = df_filtered["vid"].to_list()
        self.vid2idx = {
            vid: i for i, vid in enumerate(self.video_names)
        }

        # self.vn_vocab = 5523
        # self.matched_VNs = json.load(open("./GloFE/notebooks/openasl-v1.0/uncased_filtred_glove_VN_matched_train.json", "r"))
        # self.vn_to_idx = {}
        # with open("./GloFE/notebooks/openasl-v1.0/uncased_filtred_glove_VN_idxs.txt", "r") as f:
        #     content = f.readlines()
        #     for line in content:
        #         items = line.strip().split(" ")
        #         self.vn_to_idx[items[1]] = int(items[0])
        # vn_lens = [len(v) for _,v in self.matched_VNs.items()]
        # self.max_vns = max(vn_lens)

        if os.path.exists(preprocess_config["preprocessing_sign"]["translation_token_ids"]):
            self.translation_token_ids = [np.array([]) for _ in range(len(self.translation))]
            for t in open(preprocess_config["preprocessing_sign"]["translation_token_ids"]).readlines():
                vid, trans_ids_str = t.rstrip("\n").split("|")
                trans_ids = list(map(int, trans_ids_str.split(" ")))
                self.translation_token_ids[self.vid2idx[vid]] = np.array(trans_ids)
        else:
            pool = Pool(processes=32)
            self.translation_token_ids = [np.array([]) for _ in range(len(self.translation))]
            trans_ids_txt_list = ["" for _ in range(len(self.translation))]

            with tqdm(total=len(self.translation), desc="preprocess sign translation") as t:
                for i, trans_ids, trans_ids_txt in pool.imap_unordered(write_txt, enumerate(zip(self.video_names, self.translation, [preprocess_config] * len(self.translation)))):
                    self.translation_token_ids[i] = np.array(trans_ids)
                    trans_ids_txt_list[i] = trans_ids_txt
                    t.update(1)
            with open(preprocess_config["preprocessing_sign"]["translation_token_ids"], "w") as f:
                f.writelines(trans_ids_txt_list)

        assert len(self.translation_token_ids) == len(
            self.video_names), f"Text ids count:{len(self.translation_token_ids)}\tVid count:{len(self.video_names)}"
        all_len = np.array([len(tk)
                               for tk in self.translation_token_ids])
        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        if self.local_rank == 0:
            print(f"Max sequence length:{self.max_seq_len}")

    def __len__(self):
        return len(self.video_names)

    def read_pose_files(self, index: int):
        # MMPose 76
        body_sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        face_sample_indices = [71, 77, 85, 89] + \
                              [40, 42, 44, 45, 47, 49] + \
                              [59, 60, 61, 62, 63, 64] + [65, 66, 67, 68, 69, 70] + \
                              [50]

        # read files
        vid_name = self.video_names[index]

        file_path = os.path.join(self.feat_path, f"{vid_name}.pkl")
        with open(file_path, "rb") as f:
            pose_keypoints = pickle.load(f)  # T K(133) C

        # 23(17+6) 11 selected
        body_pose = pose_keypoints[:, body_sample_indices, :]
        hand_right = pose_keypoints[:, 91:112, :]  # 21 Keypoints
        hand_left = pose_keypoints[:, 112:, :]  # 21 Keypoints
        face = pose_keypoints[:, face_sample_indices, :]  # 23 Keypoints

        pose_tuple = (body_pose, hand_left, hand_right, face)
        pose_cated = np.concatenate(
            pose_tuple, axis=1)  # [F, 11+21+21+23=76, 3]

        # scale to [-1, 1]
        # normalization, same as pre-training, might not align with actual output shape which assumed to be [288, 384]
        pose_cated[:, :, 0:2] = 2.0 * ((pose_cated[:, :, 0:2] / 256.0) - 0.5)

        # pad pose
        T, V, C = pose_cated.shape
        # assert T == len(filenames)
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

        return pose_output, pose_length

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
        if self.phase == "train":
            raw_translation = self.translation[index]
            text_tokens, mask, token_length = self.pad_token_ids(
                index)  # [max_seq_len]
            visual_prefix, visual_length = self.read_pose_files(index)
            agx = np.random.randint(-60, 60)
            agy = np.random.randint(-60, 60)
            s = np.random.uniform(0.5, 1.5)
            # augmentation
            visual_prefix[:, :, :2] = self.rand_view_transform(
                visual_prefix[:, :, :2], agx, agy, s)[:, :, :2]
            visual_prefix[:, :, :2] = self.normalize_joints(
                visual_prefix[:, :, :2])
            assert visual_prefix.shape[0] >= self.visual_token_num, "{} {}".format(visual_prefix.shape[0], self.visual_token_num)
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            # vn_idxs, vn_len = self.get_vn(index)
            return raw_translation, text_tokens, mask, visual_prefix, token_length, visual_length  # , vn_idxs, vn_len
        elif self.phase == "test":
            raw_translation = self.translation[index]
            visual_prefix, visual_length = self.read_pose_files(index)
            visual_prefix[:, :, :2] = self.normalize_joints(
                visual_prefix[:, :, :2])
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            return raw_translation, visual_prefix, index, visual_length

    def reprocess(self, raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, idxs):
        raw_texts = [raw_texts[idx] for idx in idxs]
        text_tokens = torch.from_numpy(np.array([text_tokens[idx] for idx in idxs])).long()
        mask = torch.from_numpy(np.array([mask[idx] for idx in idxs])).float()
        visual_prefix = torch.from_numpy(np.array([visual_prefix[idx] for idx in idxs])).float()
        token_length = torch.from_numpy(np.array([token_length[idx] for idx in idxs]))
        visual_length = torch.from_numpy(np.array([visual_length[idx] for idx in idxs]))

        return raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length

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

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(raw_texts, text_tokens, mask, visual_prefix, token_length, visual_length, idx))

        return output


class UnpairedAudioSignDataset(Dataset):

    def __init__(self, filename, preprocess_config, train_config,
                 args, phase="train", split="train", partial_list_path=None):
        self.audio_dataset = AudioDataset(filename, preprocess_config, train_config)
        self.sign_dataset = SignDataset(args, preprocess_config, train_config, phase, split, partial_list_path)

        self.audio_size = len(self.audio_dataset)
        self.sign_size = len(self.sign_dataset)

        self.speaker_num = self.audio_dataset.speaker_num

        self.serial_batches = train_config["dataset"]["serial_batches"]

    def __len__(self):
        return max(len(self.audio_dataset), len(self.sign_dataset))

    def __getitem__(self, index):
        idx_audio = index % self.audio_size
        if self.serial_batches:
            idx_sign = index % self.sign_dataset
        else:
            idx_sign = random.randint(0, self.sign_size - 1)
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
