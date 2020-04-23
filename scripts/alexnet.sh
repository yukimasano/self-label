#!/bin/bash
# script for running alexnet learning on multiple gpus
device="0,1,2,3"
DIR=/tmp/ILSVRC12

# the network
ARCH="alexnet"
HC=10
NCL=3000

# the training
WORKERS=24
BS=256
NEP=400
AUG=3

# the pseudo opt
nopts=100

folder=pseudo${NCL}_${ARCH}_bs${BS}_hc${HC}-${NEP}_nopt${nopts}_augT${AUG}

EXP=./${folder}
mkdir -p ${EXP}/checkpoints/L

python3 main.py \
        --device ${device} \
        --imagenet-path ${DIR} \
        --exp ${EXP} \
        --batch-size ${BS} \
        --augs ${AUG} \
        --epochs ${NEP} \
        --nopts ${nopts} \
        --hc ${HC} \
        --arch ${ARCH} \
        --ncl ${NCL} \
        --workers ${WORKERS} \
        --comment ${EXP}  | tee -a ${EXP}/log.txt;