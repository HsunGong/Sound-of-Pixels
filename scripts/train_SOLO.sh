#!/bin/bash

OPTS=""
OPTS+="--id SOLO-single-2 "
OPTS+="--list_train data/solo/train.csv "
OPTS+="--list_val data/solo/val.csv "

# Models
OPTS+="--num_channels 64 "
# binary mask, BCE loss, weighted loss
# logscale in frequency

OPTS+="--arch_sound dprnn2 "
OPTS+="--loss upit "
OPTS+="--num_mix 2 "

# frames-related

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "
OPTS+="--instr Cello Basson Clarinet DoubleBass Flute Horn Oboe Saxophone Trumpet Violin Tuba "

# learning params

# OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--lr_steps 5 8 12 15 20 "

# display, viz
OPTS+="--disp_iter 10 "
OPTS+="--num_vis 4 "
OPTS+="--num_val 256 "

OPTS+="--batch_size_per_gpu 5 "
# OPTS+="--resume" # resume local epoch from ckpt

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u main.py $OPTS "$@"
