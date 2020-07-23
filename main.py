# System libs
import os
import random
import time

# Numerical libs
import torch
from torch.jit import optimized_execution
import torch.nn.functional as F
import numpy as np
from mir_eval.separation import bss_eval_sources

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import activate
from utils import makedirs


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame = nets
        self.crit = crit
    
    def load(self, ckpt):
        self.net_sound.load_state_dict(ckpt['sound'])
        self.net_frame.features.load_state_dict(ckpt['frame_features'])
        self.net_frame.fc.load_state_dict(ckpt['frame_fc'])


    def forward(self, gt_audios, audio_mix, frames, args):
        '''
        gt_audios: batch, num_mix, seq_len
        audio_mix: batch, seq_len
        frames : batch, num_mix, [3, H, W]
        '''
        frames.transpose_(0,1) # [2, 5, 3, 3, 224, 224]
        gt_audios.transpose_(0,1) # [2, 5, 65535]
        N = len(frames)

        # 1. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], args.img_activation)

        # 2. forward net_sound with feat-frame 
        pred_audios = self.net_sound.forward(audio_mix, feat_frames)
        pred_audios = activate(pred_audios, args.sound_activation)
        # print(pred_audios.shape, gt_audios.shape)

        # 4. loss
        err = self.crit(pred_audios, gt_audios)
        # print(err.item())#, self.crit)
        return err, [pred_audio for pred_audio in pred_audios]

def create_optimizer(nets, args, checkpoint):
    (net_sound, net_frame) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_frame.features.parameters(), 'lr': args.lr_frame},
                    {'params': net_frame.fc.parameters(), 'lr': args.lr_sound}]
    # optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer

from models.criterion import *
def build_criterion(arch):
    return eval(arch.upper() + 'Loss')()

def adjust_learning_rate(optimizer, args):
    # args.lr_sound *= 0.1
    # args.lr_frame *= 0.1
    # args.lr_synthesizer *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]

def create_dataloader(args):
    # Dataset and Loader
    dataset_train = MUSICMixDataset(
            args.list_train,
            num_frames=args.num_frames, 
            stride_frames=args.stride_frames, 
            frameRate=args.frameRate, 
            imgSize=args.imgSize, 
            audRate=args.audRate, 
            audLen=args.audLen, 
            num_mix=args.num_mix, 
            max_sample=-1, 
            dup_trainset=args.dup_trainset, 
        split='train')
    loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=True)

    dataset_val = MUSICMixDataset(
        args.list_val,
        num_frames=args.num_frames, 
        stride_frames=args.stride_frames, 
        frameRate=args.frameRate, 
        imgSize=args.imgSize, 
        audRate=args.audRate, 
        audLen=args.audLen, 
        num_mix=args.num_mix, 
        max_sample=args.num_val, 
        dup_trainset=1, 
    split='eval')

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False)

    args.epoch_iters = len(dataset_train) // args.batch_size
    return loader_train, loader_val

def main(args, checkpoint):
    # Network Builders
    from models import build_sound, build_frame
    net_sound = build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame = build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    nets = (net_sound, net_frame)
    crit = build_criterion(arch=args.loss)

    loader_train, loader_val = create_dataloader(args)

    # Wrap networks
    netWrapper = NetWrapper(nets, crit)
    if checkpoint is not None:
        print('loading at epoch:', checkpoint['epoch'])
        netWrapper.load(checkpoint)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args, checkpoint)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []}, 'val': {'epoch': [], 'err': []}
    } if checkpoint is None else checkpoint['history']

    from epoch import evaluate, train
    # Eval mode
    evaluate(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    init_epoch = 1 if checkpoint is None else checkpoint['epoch']

    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)
        loader_train.dataset.update_mix(_dilation=args.dup_trainset)
        # loader_train.dataset.update_center()

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            from utils import save_checkpoint
            save_checkpoint(nets, history, optimizer, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)
            print('LR:(sound, frame1, frame2)', get_lr(optimizer))

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.num_gpus = torch.cuda.device_count()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        # if args.binary_mask:
        #     assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
        #     args.id += '-binary'
        # else:
        #     args.id += '-ratio'
        # if args.weighted_loss:
        #     args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'eval' or args.resume:
        try:
            checkpoint = torch.load(os.path.join(args.ckpt, 'best.pth'), map_location='cpu')
            # checkpoint = os.path.join(args.ckpt, 'lastest.pth')
        except:
            checkpoint = None
    elif args.mode == 'train':
        makedirs(args.ckpt, remove=True)
        checkpoint = None
    else: raise ValueError
        

    # initialize best error with a big number
    args.best_err = float("inf")

    from utils import set_seed
    set_seed(args.seed)
    main(args, checkpoint)
