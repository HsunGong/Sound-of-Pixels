import os
import random
from dataset.base import BaseDataset
import numpy as np
import pandas as pd
import torch
from typing import List


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample,
            num_frames, stride_frames, frameRate, imgSize, 
            audRate=11025, audLen=44100,
            num_mix=2, max_sample=-1, dup_trainset=1, differ_type=True, split='train'
        ):
        super().__init__(list_sample, split,
            num_frames, stride_frames, frameRate, imgSize, 
            audRate,audLen)

        self.fps = frameRate
        self.selected_frames = (np.arange(0, self.num_frames) - self.num_frames // 2) * self.stride_frames
        
        self.num_mix = num_mix
        self.differ_type = differ_type

        self.update_mix(_dilation=dup_trainset,max_sample=max_sample)
        self.update_center()

    def __len__(self):
        return len(self.mix_idx)

    def update_mix(self, _dilation, max_sample=-1):
        '''generate a mixture index of these samples
        mix_idx : [len(dataset), idx1, idx2, ...]
        '''
        self.mix_idx = np.zeros((len(self.list_sample) * _dilation, self.num_mix), dtype=np.int)

        # types = self.list_sample['type'].unique()
        _d = self.list_sample
        for idx, record in _d.iterrows():
            self.mix_idx[idx * _dilation: (idx+1) * _dilation,0] = idx # ori idx

            _new_d = _d[_d['type'] != record['type']] if self.differ_type else _d
            for _next_idx in range(1, self.num_mix): # slow but good
                newer = pd.DataFrame(columns=_new_d.columns)
                while len(newer) < _dilation * (self.num_mix - 1):
                    # newer = newer.append(_new_d.sample((1+_dilation) * (self.num_mix - 1), axis=0, replace=False).drop_duplicates('type')) # random_state same as numpy.random.seed
                    newer = newer.append(_new_d.sample(_dilation * (self.num_mix - 1), axis=0, replace=False)) # random_state same as numpy.random.seed
                    # newer.append(_new_d.sample(_dilation * self.num_mix, axis=0)) # random_state same as numpy.random.seed
                    # newer.drop_duplicates('type', inplace=True)
                newer = newer[:_dilation * (self.num_mix - 1)].index
                self.mix_idx[idx * _dilation: (idx+1) * _dilation, _next_idx] = newer

        if max_sample > 0:
            self.mix_idx = self.mix_idx[:max_sample]
        np.random.shuffle(self.mix_idx)

    def update_center(self):
        # all_frames = self.list_sample['size']
        if self.split == 'train':
            idx_margin = max(
                int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
            # random, not to sample start and end n-frames
            self.list_sample['center_frame'] = self.list_sample['size'].apply(lambda x: 
                random.randint(idx_margin+1, x - idx_margin)
                )
        else:
            self.list_sample['center_frame'] = self.list_sample['size'].apply(lambda x: x // 2)
        
        # update audio-center
        self.list_sample['center_audio'] = self.list_sample['center_frame'].apply(lambda x: (x - 0.5) / self.fps )
        
        # center_frames = info['center']
        self.list_sample['selected_frames'] = self.list_sample['center_frame'].apply(lambda x: x + self.selected_frames)
        

    def __getitem__(self, idx):
        index = self.mix_idx[idx] # Index of videos (num_mix)
        # index[0] is the reference video-index
        info = self.list_sample.iloc[index]
        
        # Load and make
        frames = self._load_frameses(info)
        audios = self._load_audios(info)
        audios, audio_mix = self._make_mix(audios)

        ret_dict = {
            'frames' : frames,
            'audios' : audios,
            'audio_mix' : audio_mix
        }
        if self.split == 'infer': # can not use dataloader
            ret_dict['info'] = info # info.to_dict()
        return ret_dict

    def _load_frameses(self, info: pd.DataFrame) -> torch.Tensor:
        frames = info.apply(lambda x : 
            self._load_frames(x['frame'], x['selected_frames']),
        axis=1)
        return torch.stack(frames.to_list())

    def _load_audios(self, info:pd.DataFrame) -> np.ndarray:
        '''
        Warning: do not have scale function, need `_make_mix` !!!
        '''
        audios = info.apply(lambda x : 
            self._load_audio(x['audio'], x['center_audio']), 
        axis=1)
        return np.vstack(audios.to_numpy())

    def _make_mix(self, audios: np.ndarray):
        # randomize volume
        if self.split == 'train':
            scale = 1 + (0.5 - np.random.random((len(audios), 1))) * 1 
            # scale from 0.5 ~ 1.5
        else: 
            scale = 1 + (0.5 - np.random.random((len(audios), 1))) * 0
        audios *= scale
        mix = np.sum(audios, axis = 0)

        # new scale to prevent from abs(audio) > 1
        # audio[audio > 1.] = 1., audio[audio < -1.] = -1.
        scale = 0.99 / max(np.max(np.abs(audios)), np.max(np.abs(mix)))
        return (scale * audios), (scale * mix)



if __name__ == '__main__':
    import time
    st = time.perf_counter()
    # test
    # dataset = FakeMIXDataset('/slfs1/users/xg000/soundofpixel/Sound-of-Pixels/data/train.csv', 16000, 32000, 32000, 1, 2)
    # dataset = FakeMIXDataset('/slfs1/users/xg000/soundofpixel/pytorch/data/configs/8k_8/val.csv', 8000, 32000, 1, 2, _type='wav', train=False)
    dataset = MUSICMixDataset('./data/solo/val.csv',
            num_frames=1, stride_frames=1, frameRate=8, imgSize=224, 
            audRate=11025, audLen=44100, num_mix=2, max_sample=-1, dup_trainset=1, differ_type=True, split='eval')
    print('load', time.perf_counter() - st)

    print(dataset, len(dataset))
    from torch.utils.data import DataLoader

    st = time.perf_counter()
    # for idx, ((mix, refs), info) in enumerate(dataset):
    # for idx, dic in enumerate(dataset):
    for idx, dic in enumerate(DataLoader(dataset, batch_size=5)):
        # if (abs(mix) >= 1).any() or (abs(refs) >= 1).any():
        #     print(idx)
        # print(type(dic['audios']), type(dic['frames']))
        # print(type(dic['frames'][0]),type(dic['frames']))
        # raise ValueError
        # print(dic['audios'].shape, dic['frames'].shape)
        a = dic['frames']
        print(len(a), a[0].shape)
        pass
    print('time', time.perf_counter() - st)