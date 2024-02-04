#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jisheng Bai, Han Yin, Mou Wang, Haohe Liu
@email: baijs@mail.nwpu.edu.cn
# Joint Laboratory of Environmental Sound Sensing, School of Marine Science and Technology, Northwestern Polytechnical University, Xi'an, China
# Xi'an Lianfeng Acoustic Technologies Co., Ltd., China
# University of Surrey, UK
# This software is distributed under the terms of the License MIT
"""
import config
import os
import numpy as np
import pandas as pd
import glob
import librosa
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

class CAS_Dev_Dataset(object):
    def __init__(self, data_config, data_csv, is_train: bool):
        """
        :param data_config: configuration module
        :param data_csv: metadata dataframe
        :param is_train: True/False for training/validation data

        """
        self.stats_csv = data_csv
        self.root_path = data_config.dev_fea_root_path
        self.selected_scene_list = data_config.selected_scene_list
        self.tar_sr = data_config.sample_rate
        self.batch_size = data_config.batch_size
        self.clip_frames = data_config.clip_frames
        self.is_train = is_train
        self.class_2_index = data_config.class_2_index

        self.file_list = []
        self.label_list = []
        self.get_file_list()

    def get_file_list(self):
        selected_data = self.stats_csv[
            self.stats_csv["scene_label"].isin(self.selected_scene_list)
        ]

        for index, row in selected_data.iterrows():
            label_str = row["scene_label"]

            if isinstance(label_str, str):
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
                self.label_list.append(label_str)
            else:
                continue

    def get_numpy_dataset(self):
        data = []
        label = []
        for file_path in tqdm(self.file_list):
            file = np.load(file_path, allow_pickle=True)
            file = file[: self.clip_frames, :]
            data.append(file)

        for label_str in tqdm(self.label_list):
            lb = self.class_2_index[label_str]
            label.append(lb)

        data = np.asarray(data)
        label = np.asarray(label, dtype=np.int32)

        return data, label

    def get_tensordataset(self):
        data, label = self.get_numpy_dataset()
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).long()
        dataset = TensorDataset(data_tensor, label_tensor)
        if self.is_train:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        return loader


class CAS_unlabel_Dataset(object):
    def __init__(self, data_config, data_csv, fold):
        """
        :param data_config: configuration module
        :param data_csv: metadata dataframe
        :param fold: "dev"/"eval" for development/evaluation set

        """
        self.stats_csv = data_csv
        if fold == "dev":
            self.root_path = data_config.dev_fea_root_path
        else:
            self.root_path = data_config.eval_fea_root_path
        self.tar_sr = data_config.sample_rate
        self.batch_size = data_config.batch_size
        self.clip_frames = data_config.clip_frames
        self.class_2_index = data_config.class_2_index

        self.file_list = []
        self.label_list = []
        self.get_file_list()

    def get_file_list(self):
        for index, row in self.stats_csv.iterrows():
            label_str = row["scene_label"]

            if not isinstance(label_str, str):
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")

                self.file_list.append(file_path)
                self.label_list.append(label_str)
            else:
                continue

    def get_numpy_dataset(self):
        data = []
        label = []
        for file_path in tqdm(self.file_list):
            file = np.load(file_path, allow_pickle=True)
            file = file[: self.clip_frames, :]
            data.append(file)

        for label_str in tqdm(self.label_list):
            lb = label_str
            label.append(lb)

        data = np.asarray(data)
        label = np.asarray(label)

        return data, label

    def get_tensordataset(self):
        data, label = self.get_numpy_dataset()
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).float()
        dataset = TensorDataset(data_tensor, label_tensor)
        loader = DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        return loader
