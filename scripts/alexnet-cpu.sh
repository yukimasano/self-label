#!/bin/bash
# script for running alexnet learning on multiple gpus
device="0"
DIR=/tmp/ILSVRC12/train

# the network
ARCH="alexnet"
HC=10
NCL=3000

# the training
WORKERS=12
BS=256
NEP=400
AUG=3

# the pseudo opt
NOPT=100

folder=pseudo${NCL}_${ARCH}_bs${BS}_hc${HC}-${NEP}_nopt${NOPT}_augT${AUG}

EXP=./${folder}
mkdir -p ${EXP}/checkpoints/L

python3 main.py \
        --cpu \
        --device ${device} \
        --imagenet-path ${DIR} \
        --exp ${EXP} \
        --batch-size ${BS} \
        --augs ${AUG} \
        --epochs ${NEP} \
        --nopts ${NOPT} \
        --hc ${HC} \
        --arch ${ARCH} \
        --ncl ${NCL} \
        --workers ${WORKERS} \
        --comment ${EXP}  | tee -a ${EXP}/log.txt;