#!/bin/bash
DIR=/tmp/imagenet

ARCH="resnetv2"
HC=10
NCL=3000

DEVICE=3
CKPT='./self-label_models/resnet-10x3k.pth'
EXP='./eval/resnet'
mkdir -p ${EXP}

python3 eval_resnet.py \
            --hc ${HC} \
            --ncl ${NCL} \
            --arch=${ARCH} \
            --modelpath=${CKPT}\
            --datadir=${DIR}\
            --ckpt-dir=${EXP}/checkpoints-eval \
            --name=${EXP}-evallinear \
            --device=${DEVICE}