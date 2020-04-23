#!/bin/bash

type=10
# the network
hc=10       # number of heads
ncl=128     # number of clusters
arch='alexnet'

# optimization
nopts=400   # number of SK-optimizations
epochs=400  # numbers of epochs

# other
device='1'  # cuda device
bs=128      # batchsize
lr=0.03     #learning rate

dir="/scratch/local/ramdisk/yuki/data"
folder=cifar${type}/${arch}-K${ncl}_lr${lr}_bs${bs}_hc${hc}_nopt${nopts}_n${epochs}_linear_sequential
EXP=/scratch/shared/nfs1/yuki/pseudolabs/${folder}
mkdir -p ${EXP}

python3 cifar.py \
  --arch ${arch} \
  --device ${device} \
  --exp ${EXP} \
  --datadir ${dir} \
  --type ${type} \
  --batch-size ${bs} \
  --lr ${lr} \
  --nopts ${nopts} \
  --epochs ${epochs} \
  --hc ${hc} \
  --ncl ${ncl} | tee -a ${EXP}/log.txt

