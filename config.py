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
from datetime import datetime
import os

def get_timestamp():
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return timestamp

# Paths
exp_id = "baseline"
exp_timestamp = get_timestamp()
audio_root_path = "data"
dev_meta_csv_path = "./data/ICME2024_ASC_dev_label.csv"
dev_fea_root_path = "./data/feature/train"
eval_meta_csv_path = "./data/ICME2024_ASC_eval.csv"
eval_fea_root_path = "./data/feature/eval"
pretrained_model_path = "./data/pretrained/best_model.pth"
output_path = r"./log/{}_{}".format(exp_id, exp_timestamp)

os.makedirs("log", exist_ok=True)

###
selected_scene_list = [
    "Bus",
    "Airport",
    "Metro",
    "Restaurant",
    "Shopping mall",
    "Public square",
    "Urban park",
    "Traffic street",
    "Construction site",
    "Bar",
]
class_2_index = {
    "Bus": 0,
    "Airport": 1,
    "Metro": 2,
    "Restaurant": 3,
    "Shopping mall": 4,
    "Public square": 5,
    "Urban park": 6,
    "Traffic street": 7,
    "Construction site": 8,
    "Bar": 9,
}

index_2_class = {
    0: "Bus",
    1: "Airport",
    2: "Metro",
    3: "Restaurant",
    4: "Shopping mall",
    5: "Public square",
    6: "Urban park",
    7: "Traffic street",
    8: "Construction site",
    9: "Bar",
}

# Signal Processing Setting
sample_rate = 44100
clip_frames = 500
n_fft = 2048
win_length = 1764
hop_length = 882
n_mels = 64
fmin = 10
fmax = sample_rate / 2

# Model Setting
device = "cuda:0"
random_seed = 1234
train_val_ratio = 0.8
batch_size = 32
max_epoch = 20
early_stop_epoch = 10
lr = 1e-3
lr_step = 2
lr_gamma = 0.9
nhead = 8
dim_feedforward = 32
n_layers = 1
dropout = 0.1
