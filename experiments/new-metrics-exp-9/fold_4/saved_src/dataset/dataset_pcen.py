from enum import unique
import os
from os.path import join
import ast

import pandas as pd

import librosa
import numpy as np

import torch
from torch.utils.data import Dataset

import json
import torchaudio as ta

class CustomDataset(Dataset):
    def __init__(self, cfg, *, train):
        self.cfg = cfg
        self.train = train
        self.df = pd.read_csv(join(cfg['data']['dir'], cfg['data']['csv_file']))
        self.audio_dir = cfg['data']['audio_dir']

        self._setup_df()
        self.bird2id_array = {
            'indices' : [self.bird2id[x] for x in self.all_birds],
            'birds' : [self.all_birds]
        }
        with open(join(cfg['data']['dir'], 'bird_count.json'), 'r') as file:
            self.bird_count = json.load(file)
        with open(join(cfg['data']['dir'], 'scored_birds.json'), 'r') as file:
            self.scored_birds = json.load(file)

        self.scored_birds_mask = np.zeros((self.num_classes, ))
        for x in self.scored_birds:
            self.scored_birds_mask[self.bird2id[x]] = 1

        sampling_rate = self.cfg['data']['sample_rate']
        if train:
            self.aug = self.cfg['training']['train_augs']
            print(f"Dataset size before length filter : {len(self.df)}")

            mask = (self.df.length < cfg['data']['max_audio_length'] * sampling_rate) & (self.df.length > cfg['data']['min_audio_length'] * sampling_rate)
            self.df = self.df[mask].reset_index(drop=True)
            self.labels = self.labels[mask]
            print(f"Dataset size after length filter : {len(self.df)}")

        folds = self.cfg['training']['train_folds'] if train else self.cfg['training']['val_folds']

        condition = self.df.kfold.apply(lambda x : x in folds)
        self.df = self.df[condition].reset_index(drop=True)
        self.labels = self.labels[condition]
        if cfg['general']['dev']:
            self.df = self.df.iloc[:100]
            self.labels = self.labels[:100]
        print(f"Dataset length : {len(self.df)}")
        # index_name = 'train_indices.npy' if train else 'val_indices.npy'
        # indices = np.load(join(self.cfg['data']['dir'], index_name))
        # if cfg['general']['dev']:
        #     indices = np.clip(indices, 0, 99).astype(np.int32)
        # self.df = self.df.iloc[indices].reset_index(drop=True)
        # self.labels = self.labels[indices]
        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg['data']['sample_rate'],
            n_fft=cfg['data']['window_size'],
            win_length=cfg['data']['window_size'],
            hop_length=cfg['data']['hop_size'],
            f_min=cfg['data']['fmin'],
            f_max=cfg['data']['fmax'],
            pad=0,
            n_mels=cfg['data']['mel_bins'],
            power=cfg['data']['power'],
            normalized=False,
        )
        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=None)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

    def _setup_df(self):
        self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)

        ## since unique returns labels
        ## in order or appearence,
        ## in can be done as dynamic preprocessing
        unique_birds = self.df.primary_label.unique()
        secondary_labels = []
        if self.cfg['data']['use_secondary_labels']:
            for row in self.df.itertuples():
                if self.cfg['data']['treat_secondary_unique']:
                    secondary_labels.extend([f'secondary_{x}' for x in ast.literal_eval(row.secondary_labels)])
                else:
                    secondary_labels.extend(ast.literal_eval(row.secondary_labels))
        unique_secondary_birds = np.unique(secondary_labels)
        all_birds = np.concatenate([unique_birds, unique_secondary_birds])
        if self.cfg['data']['use_secondary_labels'] and not self.cfg['data']['treat_secondary_unique']:
            all_birds = np.unique(all_birds)
        all_birds = list(all_birds)
        self.all_birds = all_birds
        self.bird2id = {k:v for k, v in zip(all_birds, range(len(all_birds)))}
        self.id2bird = {k:v for k, v in zip(range(len(all_birds)), all_birds)}
        # print(self.bird2id, self.id2bird)
        print(f"Total unique birds : {len(all_birds)}")
        self.num_classes = len(all_birds)
        self.labels = np.zeros((len(self.df), len(all_birds)))
        for index, row in enumerate(self.df.itertuples()):
            current_labels = []
            current_labels.append(row.primary_label)
            if self.cfg['data']['use_secondary_labels']:
                if self.cfg['data']['treat_secondary_unique']:
                    current_labels.extend([f'secondary_{x}' for x in ast.literal_eval(row.secondary_labels)])
                else:
                    current_labels.extend(ast.literal_eval(row.secondary_labels))

            for bird in current_labels:
                if bird != "nocall":
                    bird_id = self.bird2id[bird]
                    self.labels[index, bird_id] = 1
    def __getitem__(self, index):
        sr = self.cfg['data']['sample_rate']
        crop_length = self.cfg['data']['crop_length']
        filename = self.df['filename'][index]
        label = self.labels[index]
        weight = [self.df['weight'][index]]

        if self.train:
            length = self.df.length[index]
            length_in_sec = length / sr
            duration = self.cfg['data']['crop_length']
            offset = np.random.randint(0, max(length_in_sec - crop_length, 1))
        else:
            ## take first 5 seconds of each train sample and determine if it
            ## has bird in it
            duration = self.cfg['data']['crop_length']
            offset = 0

        ## is audio length > crop_length, it will be loaded with offset
        ## otherwise, it will be paddded with silience
        audio = self._load_one(self.audio_dir, filename, offset, duration)
        audio = self._to_length(audio, crop_length * sr)

        if self.train and (self._label_in_scored_birds(label)):
            augs = self.cfg['training']['augs']

            ## random time-reverse
            if np.random.random() <= augs['time_reverse']:
                audio = np.flip(audio).copy()
            if np.random.random() <= augs['time_stretch']:
                left, right = augs['time_stretch_bounds']
                rate = np.random.uniform(left, right)
                audio = librosa.effects.time_stretch(audio, rate)
                audio = self._to_length(audio, crop_length * sr)

            ## random half-swap
            if np.random.random() <= augs['half_swap']:
                length = len(audio)
                new_audio = np.zeros_like(audio)
                ## assume length % 2 == 0
                new_audio[length // 2:] = audio[:length // 2]
                new_audio[:length // 2] = audio[length // 2:]
                audio = new_audio
                # print(audio.shape)

            audio = self.aug(samples=audio, sample_rate=sr)
        ## take 1 window less to be divisible by 5
        pcen_melspec = self._audio_to_pcen_melspec(audio)[:, :3000]
        return {
            'spec' : torch.Tensor(pcen_melspec),
            'target' : torch.Tensor(label),
            'weight' : torch.Tensor(weight)
        }

    def _to_length(self, audio, length):
        if audio.shape[0] < length:
            audio = np.pad(audio, (0, length - audio.shape[0]))
        elif audio.shape[0] > length:
            audio = audio[:length]
        return audio
        
    def _audio_to_pcen_melspec(self, audio):
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.cfg['data']['sample_rate'],
            n_fft=self.cfg['data']['window_size'],
            hop_length=self.cfg['data']['hop_size'],
            n_mels=self.cfg['data']['mel_bins'],
            fmin=self.cfg['data']['fmin'],
            fmax=self.cfg['data']['fmax'],
            power=self.cfg['data']['power'],
        )
        pcen_melspec = librosa.pcen(
            melspec * (2 ** 31),
            time_constant=0.06,
            eps=1e-6,
            gain=0.8,
            power=0.25,
            bias=10,
            sr=self.cfg['data']['sample_rate'],
            hop_length=self.cfg['data']['hop_size'],
        )
        return pcen_melspec

    def _label_in_scored_birds(self, label):
        return np.sum(self.scored_birds_mask * label) != 0

    def _load_one(self, audio_dir, filename, offset, duration):
        path = join(audio_dir, filename)
        try:
            wave, sr = librosa.load(path, sr=None, offset=offset, duration=duration)
        except:
            print(f"Failed reading record : {path}")

        return wave

    def __len__(self) -> int:
        return len(self.df)
