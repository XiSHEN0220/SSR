#!/bin/bash

DATASET="tieredImageNet"
EXPNAME="ResNet12_tieredImageNet"
FEATPTH="../ckpts/TieredImageNet/tieredImageNet_ResNet12.pth"
ARCH="ResNet12"
NITER=30000
LRDNI=1e-3
GPU=0

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 1

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 2

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 3

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 4


LRDNI=2e-3
python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 1

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 2

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 3

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 4


LRDNI=5e-3
python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 1

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 2

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 3

python few_shot_train.py  --gpu $GPU --dataset $DATASET --expName $EXPNAME --resumeFeatPth $FEATPTH --architecture $ARCH --lr-dni $LRDNI  --nbIter $NITER --nStep 4


