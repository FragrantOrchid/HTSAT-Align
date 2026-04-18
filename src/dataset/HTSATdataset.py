import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import torch, torchaudio
import pandas as pd
from util.PetersonBarneyGaussian import getMatrix as getPetersonBarneyGaussianMatrix
from util.HillenbrandGaussian import getMatrix as getHillenbrandGaussianMatrix
import logging
from joblib import Memory
memory = Memory(location='/users/u220110626/.cache', verbose=0, mmap_mode='r')
# memory.clear(warn=False)
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

    def train_dataloader(self):
        dataset =  self.HTSATsubdataset(self.train_file,self.label_vsc,self.sound_length,self.entropy_film,self.vowel_embed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # 设置 batch_size
            shuffle=True,  # 训练集需要 shuffle
            num_workers=4,  # 多线程加载数据
            pin_memory=True,  # 如果使用 GPU，可以加速数据传输
        )
    def val_dataloader(self):
        dataset = self.HTSATsubdataset(self.val_file,self.label_vsc,self.sound_length,self.entropy_film,self.vowel_embed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size*12000*14,  # 验证集 batch_size 可以相同或不同
            shuffle=False,  # 验证集不需要 shuffle
            num_workers=4,
            pin_memory=True,
        )
    class HTSATsubdataset(Dataset):
        def __init__(self,datafile,label_csv,sound_length, entropy_film: bool, vowel_embed: bool):
            self.sound_length = sound_length
            

            # 需要精准的时间映射，横向需要是以0.1s分割的，或者其倍数
            # 暂时每100ms分16份
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=32000,
                center=False,
                pad=150,
                hop_length=200, # 0.1*sampel_rate/16
                win_length=500,
                n_fft=500,
                n_mels=64
            )
            self.labels = pd.read_csv(label_csv)

            self.entropy_film=entropy_film
            self.vowel_embed=vowel_embed

            with open(datafile, 'r') as file:
                self.data = json.load(file)['data']

        def __len__(self):
            return len(self.data)
        # 这些函数，输出统一使用numpy,便于缓存
        @memory.cache(ignore=["self","index"])
        def get_target(self, index, filename):
            filelabels = self.data[index]['labels'].split(',')
            label_indexs = self.labels.loc[self.labels["mid"].isin(filelabels)]["index"].tolist()
            target = np.zeros(len(self.labels))
            for label_index in label_indexs:
                label_index = int(label_index)
                target[label_index] = 1.0
            return target
        @memory.cache
        def get_vowel(filename):
            # return torch.FloatTensor(getMatrix(filename=filename))
            return getHillenbrandGaussianMatrix(filename=filename)
        @memory.cache(ignore=["sound_length"])
        def get_log_mel(filename, sound_length):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 加载与重采样
            waveform, sr = torchaudio.load(filename, format="wav")
            waveform = waveform.to(device)
            waveform = waveform.mean(dim=0, keepdim=True)
            waveform = torchaudio.functional.resample(waveform,sr,32000)
            waveform = waveform - waveform.mean()
            # 规范声音长度到：采样率*目标长度
            current_length = waveform.shape[-1]
            if current_length < sound_length*32000:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0,sound_length*32000-current_length),
                    mode='constant',
                    value=0
                )
            elif current_length > sound_length*32000:
                waveform = waveform[...,:sound_length*32000]
            # 对数梅尔特征
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=32000,
                center=False,
                pad=150,
                hop_length=200, # 0.1*sampel_rate/16
                win_length=500,
                n_fft=500,
                n_mels=64
            ).to(device)
            return torchaudio.transforms.AmplitudeToDB()(mel_spec(waveform)).detach().cpu().numpy().copy()
        @memory.cache
        def get_entry(filename):
            # 加载与重采样
            waveform, sr = torchaudio.load(filename, format="wav")
            waveform = waveform.mean(dim=0, keepdim=True)
            waveform = torchaudio.functional.resample(waveform,sr,32000)
            waveform = waveform - waveform.mean()
            stft = torch.stft(
                input=waveform,
                n_fft=256,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True,
                window=torch.hann_window(256)
            )
            power_spectrum = torch.abs(stft) ** 2
            power_sum = torch.sum(power_spectrum)
            if power_sum < 1e-10:
                spectral_prob = torch.ones_like(power_spectrum) / power_spectrum.numel()
            else:
                spectral_prob = power_spectrum / power_sum
            log_prob = torch.log2(spectral_prob + 1e-10)
            entropy = -torch.sum(spectral_prob * log_prob)
        
        def __getitem__(self, index):
            filename = self.data[index]['wav']
            result = {}
            # 常规特征
            result["target"] = self.get_target(index=index, filename=filename)
            result["log_mel"] = self.get_log_mel(filename=filename,sound_length=self.sound_length)
            # 能量谱熵特征
            if self.entropy_film:
                result["entropy"] = self.get_entry(filename)
            # 元音特征
            if self.vowel_embed:
                result["vowel"] = self.get_vowel(filename)

            return result

