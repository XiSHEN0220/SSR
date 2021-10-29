import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import resnet12

###### --- Conv4-64 --- ######
# This arch is the same as Spyros's works: https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/ConvNet.py

###### --- ResNet12 --- ######
# ResNet12 was used in the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

##### --- WRN --- ###
# WRN is used in SIB: https://github.com/hushell/sib_meta_learn
# is also used in Spyros's work
        
        
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, has_Relu = True, has_BN = True, maxpool = True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))
        if has_BN : 
            self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        if has_Relu : 
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))
        
        if maxpool : 
            self.layers.add_module(
                'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet_4_64(nn.Module):
    def __init__(self, inputW=80, inputH=80):
        super(ConvNet_4_64, self).__init__()

        conv_blocks = []
        ## 4 blocks, each block conv + bn + relu + maxpool, with filter 64
        conv_blocks.append(ConvBlock(3, 64))
        for i in range(2):
            conv_blocks.append(ConvBlock(64, 64))
        conv_blocks.append(ConvBlock(64, 64, False)) 
        
        
        self.conv_blocks = nn.Sequential(*conv_blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        
        out = out.view(out.size(0),-1)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.droprate = dropRate
        if self.droprate > 0:
            self.dropoutLayer = nn.Dropout(p=self.droprate)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = out if self.equalInOut else x
        out = self.conv1(out)
        if self.droprate > 0:
            out = self.dropoutLayer(out)
            #out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))

        if not self.equalInOut:
            return self.convShortcut(x) + out
        else:
            return x + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            in_plances_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(block(in_plances_arg, out_planes, stride_arg, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, userelu=True, isCifar=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate) if isCifar \
                else  NetworkBlock(n, nChannels[0], nChannels[1], block, 2, dropRate)
        # 2nd block

        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True) if userelu else None
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train(self, mode=True, freeze_bn=False):
        """
        Override the default train() to freeze the BN parameters
        """
        super(WideResNet, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            print("Freezing Weight/Bias of BatchNorm2D.")
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                        
    def forward(self, x, return_spatial = False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)

        if self.relu is not None:
            out = self.relu(out)
        if return_spatial : 
            return out
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(-1, self.nChannels)

        return out


        
class dni_linear(nn.Module):
    def __init__(self, input_dims, out_dims, dni_hidden_size=1024):
        super(dni_linear, self).__init__()
        
        ## using BN achieves similar perf but Instance Norm1d introduces less parameters
        ## also instance norm1d makes the setting [Ref] X [SNN] robust to number of queries, since it doesn't reply on the batch statistic
        
        self.layer1 = nn.Sequential(
                                  nn.InstanceNorm1d(1, affine=True),
                                  nn.Linear(input_dims, dni_hidden_size),
                                  nn.ReLU(),
                                  nn.InstanceNorm1d(1, affine=True)
                                  
                                  )
        self.layer2 = nn.Sequential(
                          nn.Linear(dni_hidden_size, dni_hidden_size),
                          nn.ReLU(),
                          nn.InstanceNorm1d(1, affine=True)
                          
                          )
        self.layer3 = nn.Linear(dni_hidden_size, out_dims)
        
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.squeeze(1)
        return out





def get_featnet(architecture, inputW=80, inputH=80, dataset = 'miniImageNet'):
    # if cifar dataset, the last 2 blocks of WRN should be without stride
    isCifar = (inputW == 32) or (inputH == 32)
    if architecture == 'WRN_28_10':
        net = WideResNet(28, 10, isCifar=isCifar)
        return net, net.nChannels

    elif architecture == 'ConvNet_4_64':
        return eval(architecture)(inputW, inputH), int(64 * int(inputH/2**4) * int(inputW/2**4))

    elif architecture == 'ResNet12':
        if dataset == 'miniImageNet' or dataset == 'tieredImageNet': 
            return resnet12.resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5), int(640 * int(inputH/2**4) * int(inputW/2**4))
        else : 
            return resnet12.resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2), int(640 * (inputH/2**4) * (inputW/2**4))
        
    else:
        raise ValueError('No such feature net available!')
