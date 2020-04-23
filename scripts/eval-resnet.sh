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
            --batch-size=256 \
            --epochs=146 \
            --learning-rate=0.1 \
            --hc ${HC} \
            --ncl ${NCL} \
            --workers=8 \
            --arch=${ARCH} \
            --modelpath=${CKPT}\
            --datadir=${DIR}\
            --ckpt-dir=${EXP}/checkpoints-eval \
            --data=Imagenet \
            --comment=${EXP}-evallinear \
            --device=${DEVICE}