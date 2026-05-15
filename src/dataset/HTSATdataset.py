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
import re
import math
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
            self.data = pd.read_csv(self.datafile)
            df = pd.read_csv(label_csv)
            self.label2index = dict(zip(df["class"], df["index"]))
            # 数据处理模块
            self.wav2log_mel = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=32000,
                    center=False,
                    pad=462,
                    hop_length=100, 
                    win_length=1024,
                    n_fft=1024,
                    n_mels=128,
                    f_min=20,
                    f_max=8000
                ),
                torchaudio.transforms.AmplitudeToDB()
            )
            
            
        def __len__(self):
            return len(self.data)
        
        # (1,128,1024)
        def get_log_mel(self, index):
            filename = self.data.iloc[index]['wav']
            filename = os.path.expandvars(filename)
            start = float(self.data.iloc[index]['start'])
            end = float(self.data.iloc[index]['end'])

            waveform, sr = librosa.load(filename, sr=32000, offset=start, duration=end-start)
            target_length = self.sound_length * sr
            current_length = len(waveform)
            if current_length < target_length:
                waveform = np.pad(waveform, (0, target_length - current_length), mode='constant')
            elif current_length > target_length:
                waveform = waveform[:target_length]
            waveform = waveform - np.mean(waveform)
            waveform = torch.unsqueeze(torch.from_numpy(waveform), dim=0)

            log_mel = self.wav2log_mel(waveform)
            return log_mel.numpy().copy()

        # (20,43)
        def get_target(self, index):
            labels = self.data.iloc[index]['labels']
            labels = pd.DataFrame(eval(labels))
            start = float(self.data.iloc[index]['start'])
            target = np.zeros((20,43))
            for _, label in labels.iterrows():
                phoneme = label["phoneme"]
                match = re.search(r'(\d+)$', phoneme)
                if match:
                    target[
                        math.floor((float(label["start"])-start)*20):math.ceil((float(label["end"])-start)*20),
                        self.label2index[match.group(1)]
                    ] = 1
                    phoneme = phoneme[:match.start()]
                target[
                    math.floor((float(label["start"])-start)*20):math.ceil((float(label["end"])-start)*20),
                    self.label2index[phoneme]
                ] = 1
            return target


        def __getitem__(self, index):
            # 每个进程单独创建自己的LMDB环境
            # This is a version that won't freeze, but the env needs to be reimplemented every time.
            env = get_env(name = hashlib.sha256(self.datafile.encode("utf-8")).hexdigest())
            # 辅助变量
            # filelabels = self.data[index]['labels']
            result = {}
            
            # 特征
            result["log_mel"] = torch.tensor(
                cache(
                    env=env,
                    unique_keys=["index"]
                )(self.get_log_mel)(
                    index = index
                ),
                dtype=torch.float32
            )
            result["target"] = torch.tensor(
                cache(
                    env=env,
                    unique_keys=["index"]
                )(self.get_target)(
                    index=index
                ),
                dtype=torch.float32
            )

            env.close()
            return result

