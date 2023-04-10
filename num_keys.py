import numpy as np
import torch
import torch.nn as nn
import torchaudio   # soundfile 설치 해야 함
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
import os
from natsort import os_sorted


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SR = 16000  # sample rate
torch.set_printoptions(sci_mode=False)  # 소수점 출력을 제한

dir_path = Path.cwd() / 'padded files'   # wav data 경로
dir_path2 = Path.cwd() / 'testfile'   # wav data 경로

fp_list = list(dir_path.rglob("*.wav"))   # wav list

# csv 에서 '글자수' 반환
pin = pd.read_csv('Sample.csv', encoding='UTF8')
letter_num = pin['글자수'].values


# Dataset 만들기, index 매기기
class KeyDataset:
    def __init__(self, dir_pth, split='train'):
        self.dir_pth = Path(dir_pth)
        self.wave_ps = os_sorted(list(self.dir_pth.rglob("*.wav")),reverse=False)  # generator -> list
        self.wave_pths = []
        for i in self.wave_ps:
            self.wave_pths.append(i)
        split_pths = []
        for pth in self.wave_pths:
            if split == 'train' and int(pth.stem.split('.')[0]) <= 295:
                split_pths.append(pth)
            elif split == 'valid' and 295 < int(pth.stem.split('.')[0]) <= 300:
                split_pths.append(pth)
            elif split == 'test' and int(pth.stem.split('.')[0]) > 300:
                split_pths.append(pth)
        self.wave_pths = split_pths

        # wav indexing
        self.str2idx = {idx: string for idx, string in enumerate(letter_num)}

    def __len__(self):
        return len(self.wave_pths)

    def __getitem__(self, idx):  # idx -> input parameter
        pth = self.wave_pths[idx]
        audio_sample, sr = torchaudio.load(pth)
        return audio_sample[0, :352845], self.str2idx[idx]   # 8초로 모두 동일화. sr=48000 가정


# 80% 옳은 숫자 개수
def num_key():
    # make batch test
    valid_set = KeyDataset(dir_path2, split='valid')
    valid_loader = DataLoader(valid_set, batch_size=5, shuffle=False)
    for batch in valid_loader:
        audio, label = batch
        # for i in range(len(label)):
        #     if np.random.rand() < 0.1:
        #         label[i] = label[i] + np.random.randint(1, 3)
        #         label[i] = label[i] - np.random.randint(1, 3)
    return label
