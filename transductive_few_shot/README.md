# Transductive few-shot classification with SSR or K-reciprocal

Transductive few-shot classification with SSR or K-reciprocal



## Table of Content
* [1. Installation](#1-installation)
* [2. Training SSR](#2-training-ssr)
* [3. Evaluating SSR](#3-evaluating-ssr)
* [4. Evaluating K-reciprocal](#4-evaluating-k-reciprocal)



## 1. Installation

### 1.1. Dependencies

Install Pytorch adapted to your CUDA version : 

* [Pytorch 1.2.0 - Pytorch 1.7.0](https://pytorch.org/get-started/previous-versions/#linux-and-windows-1) 
* [Torchvision 0.4.0](https://pytorch.org/get-started/previous-versions/#linux-and-windows-1)


### 1.2. Pre-trained backbones

We use the same pre-trained features for CIFAR-FS, Mini-ImageNet as [SIB](https://github.com/hushell/sib_meta_learn).

pretrained features of tieredImageNet is not provided, so we train one with the base classification code released by [SIB](https://github.com/hushell/sib_meta_learn). 

Quick download (~400M, pretrained features on the meta-training set of CIFAR-FS, Mini-ImageNet, tieredImageNet) : 

``` Bash
cd ckpts
bash download_model.sh
```

### 1.3. Datasets

#### CIFAR-FS

We use the download link provided in [SIB](https://github.com/hushell/sib_meta_learn/tree/master/data) : 

``` Bash
cd data
bash download_cifar_fs.sh
```

The files need to be organised as :

```
./SSR/transductive_few_shot/data//data/cifar-fs/
├── val1000Episode_5_way_1_shot_15_query.json
├── train/
├── val/
└── test/
```

#### Mini-ImageNet

We use the download link provided in [SIB](https://github.com/hushell/sib_meta_learn/tree/master/data) : 

``` Bash
cd data
bash download_miniimagenet.sh
```

The files need to be organised as :

```
./SSR/transductive_few_shot/data/Mini-ImageNet/
├── val1000Episode_5_way_1_shot_10_query.json
├── val1000Episode_5_way_1_shot_15_query.json
├── val1000Episode_5_way_1_shot_20_query.json
├── train/
├── val/
└── test/
```

#### tieredImageNet

tieredImageNet can be download from [here](https://drive.google.com/file/d/1T-4NVTSa5T6CXKSRbymYLnWp_OrtF-mo/view), which is provided by [E3BM](https://github.com/yaoyao-liu/E3BM#download-resources)

The files need to be organised as :

```
./SSR/transductive_few_shot/data/tiered_imagenet/
├── val1000Episode_5_way_1_shot_15_query.json
├── train/
├── val/
└── test/
```


## 2. Training SSR


All the training command for different datasets are provided in ```./SSR/transductive_few_shot/src/run_ssr```

To run SSR on Mini-ImageNet: 

``` Bash
cd ./SSR/transductive_few_shot/src/
bash run_ssr/run_ssr_wrn_MiNiImageNet.sh  

```

## 3. Evaluating SSR

To evaluate on few-shot task with SSR, one need to run ```./SSR/transductive_few_shot/src/few_shot_test.py``` with the trained model. 

To see help option for test script: 

``` Bash
cd ./SSR/transductive_few_shot/src/
python few_shot_test.py --help

```

## 4. Evaluating K-reciprocal

To evaluate on few-shot task with k-reciprocal, one need to run ```./SSR/transductive_few_shot/src/few_shot_kreciprocal.py``` with the trained model. 

Some examples of command for (1-shot and 5-shot) can be found in ```./SSR/transductive_few_shot/src/run_krcpc/run.sh```, but one needs to set the model path for ```--ckptPth```.

To see help option for test script: 

``` Bash
cd ./SSR/transductive_few_shot/src/
python few_shot_kreciprocal.py --help

```










