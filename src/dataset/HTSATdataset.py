import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import torch, torchaudio
import pandas as pd
import logging
import librosa
import os
from util.PhonemesBinary import getMatrix as getPhonemesBinaryMatrix
from util.LMDBMemory import get_env, cache
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
            prefetch_factor = 2
        )
    class HTSATsubdataset(Dataset):
        def __init__(self,datafile,label_csv,sound_length):
            self.sound_length = sound_length
            self.datafile = datafile
            with open(datafile, 'r') as file:
                self.data = json.load(file)['data']
                
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
            self.amp2db = torchaudio.transforms.AmplitudeToDB()
            
        def __len__(self):
            return len(self.data)
        
        def get_phoneme_binary(self, index):
            filelabels = self.data[index]['labels'].split(',')
            return np.maximum.reduce([getPhonemesBinaryMatrix(filelabel) for filelabel in filelabels])
        
        def get_phoneme_binary_by_labels(self, filelabels:str):
            filelabels = filelabels.split(',')
            return np.maximum.reduce([getPhonemesBinaryMatrix(filelabel) for filelabel in filelabels])
        
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

            log_mel = self.amp2db(self.mel_spec(waveform))
            return log_mel.numpy().copy()


        def __getitem__(self, index):
            # 每个进程单独创建自己的LMDB环境

            # This is the version will freeze
            # if not hasattr(self, 'env'):
            #     self.env = get_env(location = "/users/u220110626/.cache/LMDBMemory/", name = hashlib.sha256(self.datafile.encode("utf-8")).hexdigest())
            # env = self.env

            # This is a version that won't freeze, but the env needs to be reimplemented every time.
            env = get_env(name = hashlib.sha256(self.datafile.encode("utf-8")).hexdigest())
            # 辅助变量
            filelabels = self.data[index]['labels']
            result = {}
            
            # 特征
            result["phoneme_target"] = torch.tensor(
                cache(
                    env=env,
                    unique_keys=["filelabels"]
                )(self.get_phoneme_binary_by_labels)(
                    filelabels = filelabels
                ),
                dtype=torch.float32
            )

            result["log_mel"] = torch.tensor(
                cache(
                    env=env,
                    unique_keys=["index"]
                )(self.get_log_mel)(
                    index = index
                ),
                dtype=torch.float32
            )

            env.close()
            return result

