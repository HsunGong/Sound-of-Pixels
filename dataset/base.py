import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image

from . import video_transforms as vtransforms

from typing import Union
import pandas as pd
import os.path as P

def parse_csv(filename) -> Union[pd.Series , pd.Series, pd.Series]:
    df = pd.read_csv(filename, header=None, names=["audio", "frame", "size"])
    df['size'] = pd.to_numeric(df['size'], errors='coerce').astype(np.int32)
    return df

def base_dirname(path, n):
    """Given path d, go up n dirs from d and return that path"""
    for _ in range(n):
        path = P.dirname(path)
    return P.basename(path)

class BaseDataset(torchdata.Dataset):
    @staticmethod
    def _generate(df:pd.DataFrame):
        # datas = pd.DataFrame({'path':self.list_sample})
        df['id'] = df.apply(lambda x: base_dirname(P.splitext(x['audio'])[0], 0), axis=1)
        df['type'] = df.apply(lambda x: base_dirname(P.splitext(x['audio'])[0], 1), axis=1)
        # df= df[df['type'].str.match('Cello|Bassoon')]
        df= df[df['type'].str.match('Cello|Bassoon')]
        # print(df)
        return df
        # return df[df['type'].str.match('Cello|DoubleBass')]
        # return df

    def __init__(self, list_sample, split,
            num_frames, stride_frames, frameRate, imgSize, 
            audRate,audLen,
        ):
        self.split = split

        # params
        self.num_frames = num_frames
        self.stride_frames = stride_frames
        self.frameRate = frameRate
        self.imgSize = imgSize

        # initialize video transform
        self._init_vtransform()

        self.audRate = audRate
        self.audLen = audLen
        self.audSec = 1. * self.audLen / self.audRate

        # generate data-configs
        self.list_sample = self._generate(parse_csv(list_sample))

    def __repr__(self):
        return '# samples: {}'.format(self.__len__())

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)


    def _load_frames(self, frame_dir, indexs, _type='jpg') -> torch.Tensor:
        if _type == 'jpg':
            from PIL import Image
            imgs = []
            for idx in indexs:
                imgs.append(Image.open(f'{frame_dir}/{idx}.jpg').convert('RGB'))
        else:
            raise ValueError
        return self.vid_transform(imgs)

    def _load_audio(self, path, center_timestamp, _type='wav'):
        def _read_wav(sample_rate, path:str):
            # load audio
            if _type == 'npy':
                path = path + '.npy'
                wav = np.load(path)
            elif _type == 'wav':
                import librosa
                wav, sr = librosa.core.load(path, sr=None)
                if sample_rate != sr:
                    print('warn resample', path)
                    wav = librosa.audio.resample(wav, sr, sample_rate)
            else:
                raise ValueError('No such wav type')

            # repeat if audio is too short
            if wav.shape[0] < self.audLen:
                n = self.audLen // wav.shape[0] + 1
                wav = np.tile(wav, n)
            return wav

        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        audio_raw = _read_wav(self.audRate, path)
        
        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        return audio