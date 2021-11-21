# SSR
(NeurIPS 2021) Pytorch implementation of paper "Re-ranking for image retrieval and transductivefew-shot classification"

[[Paper](https://papers.nips.cc/paper/2021/file/d9fc0cdb67638d50f411432d0d41d0ba-Paper.pdf)] [[Project webpage](http://imagine.enpc.fr/~shenx/SSR/)] [[Video]](http://imagine.enpc.fr/~shenx/SSR/ssr.mp4) [[Slide]](http://imagine.enpc.fr/~shenx/SSR/prez.pptx)

<p align="center">
<img src="https://github.com/XiSHEN0220/SSR/blob/main/fig/SSR_teaser.png" width="400px" alt="teaser">
</p>

The project is an extension work to [SIB](https://github.com/hushell/sib_meta_learn). If our project is helpful for your research, please consider citing : 

```
@inproceedings{shen2021reranking,
  title={Re-ranking for image retrieval and transductive few-shot classification},
  author={Shen, Xi and Xiao, Yang and Hu, Shell Xu, and Sbai, Othman and Aubry, Mathieu},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Table of Content
* [1. Installation](#1-installation)
* [2. Method](#2-methods-and-results)
    * [2.1 Image retrieval](#21-image-retrieval)
      * [2.1.1 SSR module](#211-ssr-module)
      * [2.1.2 Results](#212-results)    
    * [2.2 Transductive few-shot classification ](#22-transductive-few-shot-classification)
      * [2.2.1 SSR module](#221-ssr-module)
      * [2.2.2 Results](#222-results)
* [3. Acknowledgement](#3-acknowledgement)
* [4. ChangeLog](#4-changeLog)
* [5. License](#5-license)



## 1. Installation

Code is tested under **Pytorch > 1.0 + Python 3.6** environment.

Please refer to [image retrieval](https://github.com/XiSHEN0220/SSR/tree/main/image_retrieval) and [transductive few-shot classification](https://github.com/XiSHEN0220/SSR/tree/main/transductive_few_shot) to download datasets. 

## 2. Methods and Results

SSR learns updates for a similarity graph.

It decomposes the **N * N** similarity graph into **N** subgraphs 
where rows and columns of the matrix are **ordered** depending on similarities to the **subgraph reference image**.

The output of SSR is an improved similarity matrix.

<p align="center">
<img src="https://github.com/XiSHEN0220/SSR/blob/main/fig/SSR_overview.png" width="800px" alt="teaser">
</p>



### 2.1 Image retrieval 

#### 2.1.1 SSR module

Rows : the subgraph reference image (**red**) and 
the query image (**green**);

Columns : top retrieved images of the query image (**green**).
These images are ordered according to the reference image (**red**).

<p align="center">
<img src="https://github.com/XiSHEN0220/SSR/blob/main/fig/SSR_retrieval.png" width="400px" alt="teaser">
</p>

#### 2.1.2 Results 
To reproduce the results on image retrieval datasets (rOxford5k, rParis6k), please refer to [Image Retrieval](https://github.com/XiSHEN0220/SSR/tree/main/image_retrieval)

<p align="center">
<img src="https://github.com/XiSHEN0220/SSR/blob/main/fig/image_retrieval_results.png" width="600px" alt="teaser">
</p>


### 2.2 Transductive few-shot classification 

#### 2.2.1 SSR module

We illustrate our idea with an 1-shot-2way example: 

Rows: the subgraph reference image (**red**) and the support set **S**;

Columns: the support set **S** and the query set **Q**. Both **S** and **Q** are ordered according to the reference image (**red**). 

<p align="center">
<img src="https://github.com/XiSHEN0220/SSR/blob/main/fig/SSR_fewshot.png" width="400px" alt="teaser">
</p>


#### 2.2.2 Results
To reproduce the results on few-shot datasets (CIFAR-FS, Mini-ImageNet, TieredImageNet), please refer to [transductive few-shot classification](https://github.com/XiSHEN0220/SSR/tree/main/transductive_few_shot)


<p align="center">
<img src="https://github.com/XiSHEN0220/SSR/blob/main/fig/few_shot_results.png" width="800px" alt="teaser">
</p>


## 3. Acknowledgement

* The implementation of k-reciprocal is adapted from its [public code](https://github.com/zhunzhong07/person-re-ranking/tree/master/python-version)

* The implementation of few-shot training, evaluation and synthetic gradient is adapted from [SIB](https://github.com/hushell/sib_meta_learn)



## 4. ChangeLog

* **21/10/29**, model, evaluation + training released

## 5. License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on Pytorch, and uses datasets which each have their own respective licenses that must also be followed.















