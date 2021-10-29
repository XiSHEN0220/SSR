## settings of different datasets
import numpy as np
import torchvision.transforms as transforms

def dataset_setting(dataset, nSupport, nQuery=15):

    if 'miniImageNet' in dataset :
        mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean, std=std)
        trainTransform = transforms.Compose([transforms.RandomCrop(80, padding=8),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                             lambda x: np.asarray(x),
                                             transforms.ToTensor(),
                                             normalize
                                            ])
                                            
        
        valTransform = transforms.Compose([transforms.CenterCrop(80),
                                            lambda x: np.asarray(x),
                                            transforms.ToTensor(),
                                            normalize])
        

        inputW, inputH, nbCls = 80, 80, 64

        trainDir = '../data/Mini-ImageNet/train/'
        valDir = '../data/Mini-ImageNet/val/'
        testDir = '../data/Mini-ImageNet/test/'
        episodeJson = '../data/Mini-ImageNet/val1000Episode_5_way_{:d}_shot_{:d}_query.json'.format(nSupport, nQuery)
            
    ## the preprocessing is the same as https://gitlab.mpi-klsb.mpg.de/yaoyaoliu/e3bm/-/blob/inductive/dataloader/tiered_imagenet.py        
    elif 'tieredImageNet' in dataset :
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        normalize = transforms.Normalize(mean=mean, std=std)
        trainTransform = transforms.Compose([
                                             transforms.RandomResizedCrop(84),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize
                                            ])
                                            

        valTransform = transforms.Compose([ transforms.Resize([92, 92]),
                                            transforms.CenterCrop(84),
                                            transforms.ToTensor(),
                                            normalize])
                                            
        

        inputW, inputH, nbCls = 84, 84, 351

        trainDir = '../data/tiered_imagenet/train/'
        valDir = '../data/tiered_imagenet/val/'
        testDir = '../data/tiered_imagenet/test/'
        episodeJson = '../data/tiered_imagenet/val1000Episode_5_way_{:d}_shot_{:d}_query.json'.format(nSupport, nQuery)

    elif dataset == 'Cifar':
        mean = [x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]]
        std = [x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)
        trainTransform = transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                             lambda x: np.asarray(x),
                                             transforms.ToTensor(),
                                             normalize
                                            ])
        

        valTransform = transforms.Compose([lambda x: np.asarray(x),
                                           transforms.ToTensor(),
                                           normalize])
                                           
                                           
        inputW, inputH, nbCls = 32, 32, 64

        trainDir = '../data/cifar-fs/train/'
        valDir = '../data/cifar-fs/val/'
        testDir = '../data/cifar-fs/test/'
        episodeJson = '../data/cifar-fs/val1000Episode_5_way_{:d}_shot_{:d}_query.json'.format(nSupport, nQuery)

    else:
        raise ValueError('Do not support other datasets yet.')

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
