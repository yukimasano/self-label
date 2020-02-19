#!/bin/bash
MODELROOT="./self-label_models"
mkdir -p ${MODELROOT}
cd $MODELROOT

wget http://www.robots.ox.ac.uk/~vgg/research/self-label/asset/pretrained.zip
unzip pretrained.zip && rm pretrained.zip
cd ../