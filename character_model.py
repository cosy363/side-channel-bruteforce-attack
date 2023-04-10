import torch
import torch.nn as nn
import torchaudio   # soundfile 설치 해야 함
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from natsort import os_sorted


SR = 16000  # sample rate
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # device
torch.set_printoptions(sci_mode=False)  # 소수점 출력을 제한

dir_path = Path.cwd() / 'padded files'   # wav data 경로
dir_path2 = Path.cwd() / 'testfile'   # wav data 경로

# csv 읽기
pin = pd.read_csv('Sample.csv', encoding='UTF8')
letter_num = pin['변환'].values

# Dataset 만들기, index 매기기
class KeyDataset:
    def __init__(self, dir_pth, split='train'):
        self.dir_pth = Path(dir_pth)
        self.wave_ps = list(self.dir_pth.rglob("*.wav"))  # generator -> list
        self.wave_pths = []
        for i in self.wave_ps:
            self.wave_pths.append(i)

        split_pths = []
        for pth in self.wave_pths:
            if split == 'train' and int(pth.stem.split('.')[0]) <= 295:
                split_pths.append(pth)
            elif split == 'valid' and 295 < int(pth.stem.split('.')[0]) <= 300:
                split_pths.append(pth)
            elif split == 'test' and int(pth.stem.split('.')[0]) > 250:
                split_pths.append(pth)
        self.wave_pths = split_pths

        # wav indexing
        self.str2idx = {idx: string for idx, string in enumerate(letter_num)}
        self.index = np.zeros((len(letter_num), 16))
        self.index[:, 0] = 1
        for i in range(len(self.str2idx)):
            index = 0
            while index > -1:
                index = self.str2idx[i].find('!', index)
                if index > -1:
                    self.index[i, 0] = 0
                    index += 1
                    self.index[i, index] = 1
        self.index = self.index.astype(np.float32)

    def __len__(self):
        return len(self.wave_pths)

    def __getitem__(self, idx):  # idx -> input parameter
        pth = self.wave_pths[idx]
        audio_sample, sr = torchaudio.load(pth)
        return audio_sample[0, :352845], self.index[idx, :]   # 8초로 모두 동일화. sr=48000 가정


# melspectrogram 변경 후 NN 진행
class GenreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_frame = 30
        self.n_mels = 40

        self.mel_spec_converter = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, n_mels=self.n_mels,
                                                                       f_max=6000)
        self.db_converter = torchaudio.transforms.AmplitudeToDB()

        self.nn_layer = nn.Sequential(
            nn.Linear(self.n_mels * self.num_frame, 512),  # input layer. take input and return hidden representation
            nn.ReLU(),
            nn.Linear(512, 256),  # hidden layer.
            nn.ReLU(),
            nn.Linear(256, 64),  # hidden layer.
            nn.ReLU(),
            nn.Linear(64, 16),  # output layer
        )

    def forward(self, x):
        spec = self.mel_spec_converter(x)
        db_spec = self.db_converter(spec)

        sliced_spec = db_spec[:, :, db_spec.shape[2] // 2:db_spec.shape[2] // 2 + self.num_frame]
        # flattening
        sliced_spec = sliced_spec.reshape(x.shape[0], -1)

        out = self.nn_layer(sliced_spec)
        return torch.sigmoid(out)


def validate_model(model, data_loader):
    threshold = 0.15
    val_acc = 0
    with torch.no_grad():
        for batch in data_loader:
            audio, label = batch
            prediction = model(audio)
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = -1
            val_acc += (prediction == label).sum().float().item() / label.sum().float().item()
    return val_acc

def predict2(audiolist):
    train_set = KeyDataset(dir_path)
    valid_set = KeyDataset(dir_path2,split="valid")

    # make batch test
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=5, shuffle=False)

    # make NN model
    model = GenreModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.BCELoss()
    val_acc_record = []
    num_epochs = 6
    for epoch in range(num_epochs):
        print(f'epoch: {epoch}')
        for batch in train_loader:
            audio, label = batch  # x = audio / y = label
            prediction = model(audio)

            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = validate_model(model, valid_loader)
        val_acc_record.append(val_acc)

    with torch.no_grad():
        for path in audiolist:
            audio_sample, sr = torchaudio.load(path)
            audio = audio_sample[0, :352845]
            prediction = model(audio)
        
    return prediction


def predict():
    # make set
    train_set = KeyDataset(dir_path)
    valid_set = KeyDataset(dir_path,split="valid")

    # make batch test
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=5, shuffle=False)

    # make NN model
    model = GenreModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.BCELoss()
    val_acc_record = []
    num_epochs = 6
    for epoch in range(num_epochs):
        print(f'epoch: {epoch}')
        for batch in train_loader:
            audio, label = batch  # x = audio / y = label
            prediction = model(audio)

            loss = criterion(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = validate_model(model, valid_loader)
        val_acc_record.append(val_acc)

    # plt.plot(np.arange(len(val_acc_record))+1, val_acc_record)
    # plt.show()

    with torch.no_grad():
        for batch in valid_loader:
            audio, label = batch
            prediction = model(audio)
    return prediction

