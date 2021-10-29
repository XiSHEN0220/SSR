#!/bin/bash

############# CIFAR + Conv
DATASET="Cifar"
FEATPTH="../ckpts/CIFAR-FS/Conv4_64_Cos_netFeatBest.pth"
ARCH="ConvNet_4_64"
GPU=3
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

############# miniImageNet + Conv
DATASET="miniImageNet"
FEATPTH="../ckpts/MiniImageNet/Conv4_64_Cos_netFeatBest.pth"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

############# tieredImageNet + Conv
DATASET="tieredImageNet"
FEATPTH="../ckpts/TieredImageNet/tieredImageNet_ConvNet.pth"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH








############# CIFAR + ResNet12
DATASET="Cifar"
FEATPTH="../ckpts/CIFAR-FS/ResNet12_Cos_netFeatBest.pth"
ARCH="ResNet12"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

############# miniImageNet + ResNet12
DATASET="miniImageNet"
FEATPTH="../ckpts/MiniImageNet/ResNet12_Cos_netFeatBest.pth"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

############# tieredImageNet + ResNet12
DATASET="tieredImageNet"
FEATPTH="../ckpts/TieredImageNet/tieredImageNet_ResNet12.pth"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH











############# CIFAR + WRN
DATASET="Cifar"
FEATPTH="../ckpts/CIFAR-FS/WRN_Cos_netFeatBest62.561.pth"
ARCH="WRN_28_10"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

############# miniImageNet + WRN
DATASET="miniImageNet"
FEATPTH="../ckpts/MiniImageNet/WRN_Cos_netFeatBest64.653.pth"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

############# tieredImageNet + WRN
DATASET="tieredImageNet"
FEATPTH="../ckpts/TieredImageNet/tieredImageNet_WRN.pth"
nSupport=1

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH

nSupport=5

python few_shot_kreciprocal.py --gpu $GPU --dataset $DATASET --resumeFeatPth $FEATPTH --nSupport $nSupport --architecture $ARCH


