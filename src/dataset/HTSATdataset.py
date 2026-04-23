import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import torch, torchaudio
import pandas as pd
from util.PetersonBarneyGaussian import getMatrix as getPetersonBarneyGaussianMatrix
from util.HillenbrandGaussian import getMatrix as getHillenbrandGaussianMatrix
from util.HillenbrandGaussianPCA import getMatrix as getHillenbrandGaussianPCAMatrix
import logging
import librosa
from joblib import Memory
import psutil
import os
memory = Memory(location='/users/u220110626/.cache', verbose=0, mmap_mode='r')

class HTSATdataset(pl.LightningDataModule):
    # sound_length 单位为秒
    def __init__(self, train_file, val_file, label_csv, sound_length: int, batch_size, entropy_film: bool, vowel_embed: bool):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.label_vsc = label_csv
        self.sound_length = sound_length
        self.batch_size = batch_size
        self.entropy_film=entropy_film
        self.vowel_embed=vowel_embed
        self.n_proc = int(os.environ.get("N_PROC","4"))
        print(f"n_proc = {self.n_proc}")

        
    def train_dataloader(self):
        dataset =  self.HTSATsubdataset(self.train_file,self.label_vsc,self.sound_length,self.entropy_film,self.vowel_embed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # 设置 batch_size
            shuffle=True,  # 训练集需要 shuffle
            num_workers=self.n_proc*3,  # 多线程加载数据
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True  # 如果使用 GPU，可以加速数据传输
        )
    def val_dataloader(self):
        dataset = self.HTSATsubdataset(self.val_file,self.label_vsc,self.sound_length,self.entropy_film,self.vowel_embed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # 验证集 batch_size 可以相同或不同
            shuffle=False,  # 验证集不需要 shuffle
            num_workers=self.n_proc*3,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True
        )
    class HTSATsubdataset(Dataset):
        def __init__(self,datafile,label_csv,sound_length, entropy_film: bool, vowel_embed: bool):
            self.sound_length = sound_length
            
            # 需要精准的时间映射，横向需要是以0.1s分割的，或者其倍数
            # 暂时每100ms分16份
            self.labels = pd.read_csv(label_csv)

            self.entropy_film=entropy_film
            self.vowel_embed=vowel_embed

            with open(datafile, 'r') as file:
                self.data = json.load(file)['data']

        def __len__(self):
            return len(self.data)
        # 这些函数，输出统一使用numpy,便于缓存
        @memory.cache(ignore=["data","labels","index"])
        def get_target(data, labels, index, filename):
            filelabels = data[index]['labels'].split(',')
            label_indexs = labels.loc[labels["mid"].isin(filelabels)]["index"].tolist()
            target = np.zeros(len(labels))
            for label_index in label_indexs:
                label_index = int(label_index)
                target[label_index] = 1.0
            return target
        
        @memory.cache
        def get_vowel(filename):
            return getHillenbrandGaussianPCAMatrix(filename=filename)
        
        @memory.cache(ignore=["waveform"])
        def get_log_mel(filename, waveform):
            waveform = torch.unsqueeze(torch.from_numpy(waveform), dim=0)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=32000,
                center=False,
                pad=150,
                hop_length=200, # 0.1*sampel_rate/16
                win_length=500,
                n_fft=500,
                n_mels=64
            )
            log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec(waveform))
            return log_mel.numpy().copy()
        # 如效果不佳，改用torchaudio的算法
        @memory.cache(ignore=["waveform"])
        def get_entropy(filename, waveform):
            if np.max(np.abs(waveform)) < 1e-6:  # 静音检测
                return np.array(0.0)  # 静音时返回 0 或其他默认值
            S = librosa.stft(waveform, n_fft=256)
            power_spectrum = np.abs(S) ** 2
            power_sum = np.sum(power_spectrum)
            if power_sum < 1e-10:
                spectral_prob = np.ones_like(power_spectrum) / len(power_spectrum)
            else:
                spectral_prob = power_spectrum / power_sum
            entropy = -np.sum(spectral_prob * np.log2(spectral_prob + 1e-10))
            return entropy
        
        def __getitem__(self, index):
            filename = self.data[index]['wav']
            # 波形数据与规则化长度
            waveform, sr = librosa.load(filename, sr=32000)
            
            target_length = self.sound_length * 32000
            current_length = len(waveform)
            if current_length < target_length:
                waveform = np.pad(waveform, (0, target_length - current_length), mode='constant')
            elif current_length > target_length:
                waveform = waveform[:target_length]
            waveform = waveform - np.mean(waveform)
            result = {}
            # 常规特征
            result["target"] = torch.tensor(self.get_target(data=self.data,labels=self.labels,index=index, filename=filename), dtype=torch.float32)  
            result["log_mel"] = torch.tensor(self.get_log_mel(filename=filename,waveform=waveform), dtype=torch.float32) 
            # 能量谱熵特征
            if self.entropy_film:
                result["entropy"] = torch.tensor(self.get_entropy(filename=filename,waveform=waveform), dtype=torch.float32) 
            # 元音特征
            if self.vowel_embed:
                result["vowel"] = torch.tensor(self.get_vowel(filename), dtype=torch.float32) 
                # result["vowel"] = torch.tensor(getHillenbrandGaussianPCAMatrix(filename), dtype=torch.float32) 

            return result

