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
import os
from util.PhonemesBinary import getMatrix as getPhonemesBinaryMatrix
from util.LMDBMemory import LMDBMemory
import hashlib
class HTSATdataset(pl.LightningDataModule):
    # sound_length 单位为秒
    def __init__(self, train_file, val_file, label_csv, sound_length: int, batch_size):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.label_vsc = label_csv
        self.sound_length = sound_length
        self.batch_size = batch_size
        self.n_proc = int(os.environ.get("N_PROC","4"))
        print(f"n_proc = {self.n_proc}")

        
    def train_dataloader(self):
        dataset =  self.HTSATsubdataset(self.train_file,self.label_vsc,self.sound_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # 设置 batch_size
            shuffle=True,  # 训练集需要 shuffle
            num_workers=self.n_proc,  # 多线程加载数据
            pin_memory=True,  # 如果使用 GPU，可以加速数据传输
            prefetch_factor = 2
        )
    def val_dataloader(self):
        dataset = self.HTSATsubdataset(self.val_file,self.label_vsc,self.sound_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # 验证集 batch_size 可以相同或不同
            shuffle=False,  # 验证集不需要 shuffle
            num_workers=self.n_proc,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor = 2
        )
    class HTSATsubdataset(Dataset):
        def __init__(self,datafile,label_csv,sound_length):
            self.sound_length = sound_length
            
            # 需要精准的时间映射，横向需要是以0.1s分割的，或者其倍数
            # 暂时每100ms分16份
            self.labels = pd.read_csv(label_csv)

            with open(datafile, 'r') as file:
                self.data = json.load(file)['data']
                
                        # 这个步骤移动到init TODO
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=32000,
                center=False,
                pad=150,
                hop_length=200, # 0.1*sampel_rate/16
                win_length=500,
                n_fft=500,
                n_mels=64,
                f_min=80,
                f_max=8000
            )    
            
            self.memory = LMDBMemory(location="/users/u220110626/.cache/LMDBMemory/", name = hashlib.sha256(datafile.encode("utf-8")).hexdigest(), len=len(self.data))
            self.get_log_mel_with_cache = self.memory.cache(key = "log_mel")(self.get_log_mel)
            # self.get_phoneme_binary_with_cache = self.memory.cache(key = "phoneme_binary")(self.get_phoneme_binary)

        def __len__(self):
            return len(self.data)
        
        def get_phoneme_binary(self, index):
            filelabels = self.data[index]['labels'].split(',')
            return np.maximum.reduce([getPhonemesBinaryMatrix(filelabel) for filelabel in filelabels])

        def get_target(self, index):
            filelabels = self.data[index]['labels'].split(',')
            label_indexs = self.labels.loc[self.labels["mid"].isin(filelabels)]["index"].tolist()
            target = np.zeros(len(self.labels))
            for label_index in label_indexs:
                label_index = int(label_index)
                target[label_index] = 1.0
            return target

        def get_log_mel(self, index):
            filename = self.data[index]['wav']
            waveform, sr = librosa.load(filename, sr=32000)
            target_length = self.sound_length * 32000
            current_length = len(waveform)
            if current_length < target_length:
                waveform = np.pad(waveform, (0, target_length - current_length), mode='constant')
            elif current_length > target_length:
                waveform = waveform[:target_length]
            waveform = waveform - np.mean(waveform)
            waveform = torch.unsqueeze(torch.from_numpy(waveform), dim=0)

            log_mel = torchaudio.transforms.AmplitudeToDB()(self.mel_spec(waveform))
            return log_mel.numpy().copy()


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
            result["word_target"] = torch.tensor(self.get_target(index), dtype=torch.float32)
            result["phoneme_target"] = torch.tensor(self.get_phoneme_binary(index))
            result["log_mel"] = torch.tensor(self.get_log_mel_with_cache(index), dtype=torch.float32)

            return result

