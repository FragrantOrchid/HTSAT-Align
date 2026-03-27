import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
import torchaudio
import csv
import numpy as np
import torch
import pandas as pd
class HTSATdataset(pl.LightningDataModule):
    # sound_length 单位为秒
    def __init__(self, train_file, val_file, label_csv, sound_length: int):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.label_vsc = label_csv
        self.sound_length = sound_length

    def train_dataloader(self):
        dataset =  self.HTSATsubdataset(self.train_file,self.label_vsc,self.sound_length)
        return DataLoader(
            dataset,
            batch_size=100,  # 设置 batch_size
            shuffle=True,  # 训练集需要 shuffle
            num_workers=4,  # 多线程加载数据
            pin_memory=True,  # 如果使用 GPU，可以加速数据传输
        )
    def val_dataloader(self):
        dataset = self.HTSATsubdataset(self.val_file,self.label_vsc,self.sound_length)
        return DataLoader(
            dataset,
            batch_size=100,  # 验证集 batch_size 可以相同或不同
            shuffle=False,  # 验证集不需要 shuffle
            num_workers=4,
            pin_memory=True,
        )
    class HTSATsubdataset(Dataset):
        def __init__(self,datafile,label_csv,sound_length):
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

            with open(datafile, 'r') as file:
                self.data = json.load(file)['data']

        def __len__(self):
            return len(self.data)
        #TODO 后期改成pandas来提供查找操作
        def __getitem__(self, index):
            filename = self.data[index]['wav']
            filelabels = self.data[index]['labels']

            waveform, sr = torchaudio.load(filename, format="wav")
            waveform = waveform.mean(dim=0, keepdim=True)
            waveform = torchaudio.functional.resample(waveform,sr,32000)
            waveform = waveform - waveform.mean()
            # 规范声音长度到：采样率*目标长度
            current_length = waveform.shape[-1]
            if current_length < self.sound_length*32000:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0,self.sound_length*32000-current_length),
                    mode='constant',
                    value=0
                )
            elif current_length > self.sound_length*32000:
                waveform = waveform[...,:self.sound_length*32000]
            # 特征提取
            log_mel = torchaudio.transforms.AmplitudeToDB()(self.mel_spec(waveform))
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform=waveform,
                sample_frequency=32000,
                num_mel_bins=64,
                frame_length=150.0/16,
                frame_shift=100.0/16,
                window_type="hanning",
                snip_edges=False
            )
            fbank = fbank.permute(1,0).unsqueeze(0)

            label_index = self.labels.loc[self.labels["mid"] == filelabels].iloc[0]["index"]
            label_index = int(label_index)
            target = np.zeros(len(self.labels))
            target[label_index] = 1.0
            target = torch.FloatTensor(target)# .unsqueeze(0)
            
            return {"log_mel"   :   log_mel, # Tendor shape torch.Size([1, 64, 800])
                    "fbank"     :   fbank,
                    "target"    :   target}

