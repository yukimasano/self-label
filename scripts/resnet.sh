#!/bin/bash
# script for running resnet learning on multiple gpus
device="0,1,2,3" # we need around 46GB of GPU memory for storing resnet activations in float64. half for f32
DIR=/tmp/ILSVRC12

# the network
ARCH="resnetv2"
hc=10
ncl=3000

# the training
WORKERS=24
bs=256
nepochs=400
augs=3

# the pseudo opt
nopts=100

folder=pseudo${ncl}_${ARCH}_bs${bs}_hc${hc}-${nepochs}_nopt${nopts}_augT${augs}

EXP=./${folder}
mkdir -p ${EXP}/checkpoints/L

python3 main.py \
        --cpu \
        --device ${device} \
        --imagenet-path ${DIR} \
        --exp ${EXP} \
        --batch-size ${bs} \
        --augs ${augs} \
        --epochs ${nepochs} \
        --nopts ${nopts} \
        --hc ${hc} \
        --arch ${ARCH} \
        --ncl ${ncl} \
        --workers ${WORKERS} \
        --comment ${EXP}  | tee -a ${EXP}/log.txt;
