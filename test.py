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
import pandas as pd
import os
import numpy as np
import torch
from dataset import CAS_unlabel_Dataset
from models.baseline import SE_Trans, ModelTester

def setup_seed(seed):
     '''
     :param seed: random seed
     '''
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def gen_result(config):
    '''
    :param config: configuration module
    '''
    print('========== Generate eval result ==========')
    os.makedirs(config.output_path, exist_ok=True)
    if not os.path.exists(os.path.join(config.output_path, "best_model_1.pth")):
        print('========== Second model does not exist ==========')
        return
    print('== Loading eval dataset ==')
    eval_meta_csv = pd.read_csv(config.eval_meta_csv_path, index_col=False)
    eval_unlabeled_dataset = CAS_unlabel_Dataset(config, eval_meta_csv, "eval")
    eval_unlabeled_dataloader = eval_unlabeled_dataset.get_tensordataset()

    print('== Loading model ckpt ==')
    model = SE_Trans(frames=config.clip_frames, bins=config.n_mels, 
                     class_num=len(config.class_2_index.keys()), nhead=config.nhead, 
                     dim_feedforward=config.dim_feedforward, 
                     n_layers=config.n_layers, dropout=config.dropout)
    
    model = model.to(config.device)
    ckpt = torch.load(os.path.join(config.output_path, "best_model_1.pth"))
    model.load_state_dict(ckpt['model_state_dict'])
    print('== Loading model params done ==')
    
    predicter = ModelTester(model, eval_unlabeled_dataloader, config.device)
    eval_results = predicter.predict()
    
    for index, row in eval_meta_csv.iterrows():
        eval_label_index = eval_results[index][0]
        eval_meta_csv.loc[index, "scene_label"] = config.index_2_class[eval_label_index]

    eval_meta_csv_save_path = os.path.join(config.output_path, "eval_results_1.csv")
    eval_meta_csv.to_csv(eval_meta_csv_save_path, index=False)
    print('== Saving eval results to {} =='.format(eval_meta_csv_save_path))
    
if __name__=="__main__":
    
    setup_seed(config.random_seed)
    gen_result(config)
