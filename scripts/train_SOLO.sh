#!/bin/bash

OPTS=""
OPTS+="--id SOLO-single "
OPTS+="--list_train data/solo/train.csv "
OPTS+="--list_val data/solo/val.csv "

# Models
OPTS+="--arch_sound dprnn2 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
# binary mask, BCE loss, weighted loss
# logscale in frequency
OPTS+="--loss UPIT "
OPTS+="--num_mix 2 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--workers 48 "
OPTS+="--dup_trainset 3 "

OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 20 30 40 60 80 "

# display, viz
OPTS+="--disp_iter 10 "
OPTS+="--num_vis 4 "
OPTS+="--num_val 256 "

OPTS+="--batch_size_per_gpu 5 "
OPTS+="--resume"

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u main.py $OPTS "$@"
