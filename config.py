#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jisheng Bai, Han Yin, Mou Wang, Haohe Liu
@email: baijs@mail.nwpu.edu.cn
# Joint Laboratory of Environmental Sound Sensing, School of Marine Science and Technology, Northwestern Polytechnical University, Xi’an, China
# Xi'an Lianfeng Acoustic Technologies Co., Ltd., China
# University of Surrey, UK
# This software is distributed under the terms of the License MIT
"""

# Paths
exp_id = 1
audio_root_path = ''
dev_meta_csv_path = './metadata/ICME2024_ASC_dev_label_20240126.csv'
dev_fea_root_path = './feature/train'
eval_meta_csv_path = './metadata/ICME2024_ASC_eval_20240126.csv'
eval_fea_root_path = './feature/eval'
pretrained_model_path = './pretrained/best_model.pth'
output_path = r'./exp_{}'.format(exp_id)

###
selected_scene_list = ['Bus', 'Airport', 'Metro', 'Restaurant', 
               'Shopping mall', 'Public square', 'Urban park',
               'Traffic street', 'Construction site', 'Bar']
class_2_index = { 'Bus': 0, 'Airport': 1, 'Metro': 2, 'Restaurant': 3, 
               'Shopping mall': 4, 'Public square': 5, 'Urban park': 6,
               'Traffic street': 7, 'Construction site': 8, 'Bar': 9 }

index_2_class = { 0: 'Bus', 1: 'Airport', 2: 'Metro', 3: 'Restaurant', 
               4: 'Shopping mall', 5: 'Public square', 6: 'Urban park',
               7: 'Traffic street', 8: 'Construction site', 9: 'Bar'}

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









