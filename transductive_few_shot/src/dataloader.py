import os
import torch
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import json

from torchvision import transforms
from torchvision.datasets import ImageFolder


def PilLoaderRGB(imgPath) :
    return Image.open(imgPath).convert('RGB')



class EpisodeSampler():
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nClsEpisode = 5.
    :param string imgDir: image directory, each category is in a sub file;
    :param int nClsEpisode: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param int nQuery: number of query examples;
    :param transform: image transformation/data augmentation;
    :param bool useGPU: whether to use gpu or not;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    """
    def __init__(self, imgDir, nClsEpisode, nSupport, nQuery, transform, useGPU, inputW, inputH):
        self.imgDir = imgDir
        self.clsList = os.listdir(imgDir)
        self.nClsEpisode = nClsEpisode
        self.nSupport = nSupport
        self.nQuery = nQuery
        self.transform = transform
        

        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(nClsEpisode, nSupport, 3, inputW, inputH)
        self.tensorQuery = floatType(nClsEpisode * nQuery, 3, inputW, inputH)
        self.labelQuery = intType(nClsEpisode * nQuery)
        self.imgTensor = floatType(3, inputW, inputH)
        self.useGPU = useGPU

    def getEpisode(self):
        """
        Return an episode
        :return dict: {'SupportTensor': nCls x nSupport x 3 x H x W,
                       'QueryTensor': nQuery x 3 x H x W,
                       'QueryLabel': nQuery}
        """
        
        # select nClsEpisode from clsList
        clsEpisode = np.random.choice(self.clsList, self.nClsEpisode, replace=False)
        
        # labels {0, ..., nClsEpisode-1}
        for i in range(self.nClsEpisode) :
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i
            
        for i, cls in enumerate(clsEpisode) :
            clsPath = os.path.join(self.imgDir, cls)
            imgList = os.listdir(clsPath)
            # in total nQuery+nSupport images from each class
            imgCls = np.random.choice(imgList, self.nSupport + self.nQuery , replace=False)

            for j in range(self.nSupport) :
                img = imgCls[j]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i, j] = self.imgTensor.copy_(self.transform(I))
                

            for j in range(self.nQuery) :
                img = imgCls[j + self.nSupport]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))
                

        ## Random permutation. Though this is not necessary in our approach
        permQuery = torch.randperm(self.nClsEpisode * self.nQuery)
        permQuery = permQuery.cuda() if self.useGPU else permQuery

        return {'SupportTensor':self.tensorSupport,
                'QueryTensor':self.tensorQuery.index_select(dim=0, index=permQuery),
                'QueryLabel':self.labelQuery.index_select(dim=0, index=permQuery)
                }


class ValImageFolder(data.Dataset):
    def __init__(self, episodeJson, imgDir, inputW, inputH, valTransform, useGPU):
        with open(episodeJson, 'r') as f :
            self.episodeInfo = json.load(f)

        self.imgDir = imgDir
        self.nEpisode = len(self.episodeInfo)
        self.nClsEpisode = len(self.episodeInfo[0]['Support'])
        self.nSupport = len(self.episodeInfo[0]['Support'][0])
        self.nQuery = len(self.episodeInfo[0]['Query'][0])
        self.transform = valTransform
        
        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.useGPU = useGPU
        self.tensorSupport = floatType(self.nClsEpisode, self.nSupport, 3, inputW, inputH)
        self.tensorQuery = floatType(self.nClsEpisode * self.nQuery, 3, inputW, inputH)
        self.labelQuery = intType(self.nClsEpisode * self.nQuery)

        self.imgTensor = floatType(3, inputW, inputH)
        for i in range(self.nClsEpisode) :
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i
            
        


    def __getitem__(self, index):
        for i in range(self.nClsEpisode) :
            for j in range(self.nSupport) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Support'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i, j] = self.imgTensor.copy_(self.transform(I))
                
            for j in range(self.nQuery) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Query'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))
                
        
        ## Random permutation. Though this is not necessary in our approach
        permQuery = torch.randperm(self.nClsEpisode * self.nQuery)
        permQuery = permQuery.cuda() if self.useGPU else permQuery
        return {'SupportTensor':self.tensorSupport,
                'QueryTensor':self.tensorQuery.index_select(dim=0, index=permQuery),
                'QueryLabel':self.labelQuery.index_select(dim=0, index=permQuery)
                }

    def __len__(self):
        return self.nEpisode


def ValLoaderEpisode(episodeJson, imgDir, inputW, inputH, valTransform, useGPU) :
    dataloader = data.DataLoader(ValImageFolder(episodeJson, imgDir, inputW, inputH,
                                                valTransform, useGPU),
                                 shuffle=False)
    return dataloader

