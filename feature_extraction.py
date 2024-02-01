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


import numpy as np
import scipy
import librosa
import os
from tqdm import tqdm
import pandas as pd
import glob


def gen_mel_features(
    data,
    sr,
    n_fft,
    hop_length,
    win_length,
    n_mels,
    fmin,
    fmax,
    window="hann",
    logarithmic=True,
):
    """
    :param data: input waveform
    :param sr: sampling rate
    :param n_fft: FFT samples
    :param hop_length: frame move samples
    :param win_length: window length
    :param n_mels: number of mel bands
    :param fmin: minimum frequency
    :param fmax: maximum frequency
    :param window: window type

    """
    eps = np.spacing(1)
    if window == "hann":
        window = scipy.signal.hann(win_length, sym=False)
    else:
        window = scipy.signal.hamming(win_length, sym=False)
    spectrogram = np.abs(
        librosa.stft(
            data + eps,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            window=window,
        )
    )
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False
    )

    mel_spectrogram = np.dot(mel_basis, spectrogram)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    return log_mel_spectrogram.T


def save_features(config, fold):
    """
    :param config: configuration module
    :param fold: "dev"/"eval" for development/evaluation set
    """
    print("========== Generate Feature for {} ==========".format(fold))
    if os.path.exists(config.audio_root_path):
        if fold == "dev":
            meta_csv = pd.read_csv(config.dev_meta_csv_path)
            feature_root_path = config.dev_fea_root_path
        elif fold == "eval":
            meta_csv = pd.read_csv(config.eval_meta_csv_path)
            feature_root_path = config.eval_fea_root_path
        os.makedirs(feature_root_path, exist_ok=True)
        # extract acoustic features
        print("=== Extraction Begin ===")
        with tqdm(total=len(meta_csv)) as pbar:
            for index, row in meta_csv.iterrows():
                filename = row["filename"]
                feature_save_path = os.path.join(feature_root_path, filename + ".npy")
                if os.path.exists(feature_save_path):
                    pbar.update(1)
                    continue
                
                filepath_str = os.path.join(
                    config.audio_root_path, "*" + filename + "*.wav"
                )
                # print(glob.glob(filepath_str))
                audio_path = glob.glob(filepath_str)[0]
                audio, _ = librosa.load(audio_path, sr=config.sample_rate)

                feature = gen_mel_features(
                    audio,
                    config.sample_rate,
                    config.n_fft,
                    config.hop_length,
                    config.win_length,
                    config.n_mels,
                    config.fmin,
                    config.fmax,
                )
                
                np.save(feature_save_path, feature, allow_pickle=True)
                pbar.update(1)
                # break
        print("========== End ==========")
    else:
        print("========== {} dataset does not exist ==========".format(fold))


if __name__ == "__main__":
    import config

    config.audio_root_path = "data/ICME2024_GC_ASC_dev"
    save_features(config, fold="dev")
    config.audio_root_path = "data/ICME2024_GC_ASC_eval"
    save_features(config, fold="eval")
