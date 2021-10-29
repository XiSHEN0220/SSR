import torch
import torch.nn as nn
import random
import itertools
import json
import os

import networks # get_networks
import ssr # SSR
import dataset # get datase setting
import dataloader # train / val dataloader
import utils # other functions

import argparse
import sys


def test(dataloader, netSSR, netFeat, nEpisode, device, logger, nFeat) :

    nEpisode = nEpisode
    logger.info('\n\nTest mode: randomly sample {:d} episodes...'.format(nEpisode))

    episodeAccLog = []
    top1 = utils.AverageMeter()

    for batchIdx in range(nEpisode):
        data = dataloader.getEpisode()
        data = utils.to_device(data, device)

        SupportTensor = data['SupportTensor']
        QueryTensor = data['QueryTensor']
        QueryLabel = data['QueryLabel']


        with torch.no_grad() :
            nb_cls, nb_supp = SupportTensor.size()[0], SupportTensor.size()[1]
            nb_query = QueryTensor.size()[0]
            SupportFeat = netFeat(SupportTensor.contiguous().view(-1, 3, inputW, inputH))
            QueryFeat = netFeat(QueryTensor.contiguous().view(-1, 3, inputW, inputH))

            SupportFeat, QueryFeat = SupportFeat.contiguous().view(nb_cls, nb_supp, nFeat), QueryFeat.view(nb_query, nFeat)

        SupportFeat = SupportFeat.requires_grad_(True)
        QueryFeat = QueryFeat.requires_grad_(True)

        clsScore = netSSR(SupportFeat, QueryFeat)

        QueryLabel = QueryLabel.view(-1)
        acc1 = utils.accuracy(clsScore, QueryLabel, topk=(1,))
        top1.update(acc1[0].item(), clsScore.size()[0])

        episodeAccLog.append(acc1[0].item())

        if batchIdx % 500 == 499 :
            msg = 'ID {:d}, Top1 SSR : {:.3f}%'.format(batchIdx, top1.avg)
            print (msg)

    mean, ci95 = utils.getCi(episodeAccLog)
    msg = '\t\t\t-->\t\tFinal Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95)
    logger.info(msg)

    return mean, ci95



#############################################################################################
## Parameters
parser = argparse.ArgumentParser(description='Test')

parser.add_argument('--nEpisode', type = int, default = 2000, help='nb episode')
parser.add_argument('--gpu', type=int, help='which gpu?')
parser.add_argument('--ckptPth', type = str, help='resume Path')
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'Cifar', 'tieredImageNet'], help='Which dataset? Should modify normalization parameter')
parser.add_argument('--dniHiddenSize', type=int, default=4096, help='dni hidden size')


# Validation
parser.add_argument('--nFeat', type = int, default=640, help='feature dimension')
parser.add_argument('--nSupport', type = int, default=1, help='support set, nb of shots')
parser.add_argument('--nQuery', type = int, default=15, help='query set, nb of queris')
parser.add_argument('--nClsEpisode', type = int, default=5, help='nb of classes in each set')

parser.add_argument('--architecture', type = str, default='WRN_28_10', choices=['WRN_28_10', 'ConvNet_4_64', 'ResNet12'], help='arch')
parser.add_argument('--nStep', type = int, help='nb of steps')
parser.add_argument('--lr-dni', type=float, default=1e-3, help='learning rate for dni')


args = parser.parse_args()
print (args)

# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda')

logPath = os.path.join(args.ckptPth, 'logs')
logger = utils.get_logger(logPath, 'test')

#############################################################################################
## datasets
_, valTransform, inputW, inputH, \
        trainDir, valDir, testDir, episodeJson, nbCls = \
        dataset.dataset_setting(args.dataset, args.nSupport)

args.inputW = inputW
args.inputH = inputH

testLoader = dataloader.EpisodeSampler(imgDir = testDir,
                                       nClsEpisode = args.nClsEpisode, ## always eval 5-way
                                       nSupport = args.nSupport,
                                       nQuery = args.nQuery,
                                       transform = valTransform,
                                       useGPU = args.gpu is not None,
                                       inputW = inputW,
                                       inputH = inputH)


#############################################################################################
## Networks
netFeat, args.nFeat = networks.get_featnet(args.architecture, inputW, inputH, args.dataset)


netSSR = ssr.SSR(feat_dim = args.nFeat,
                 q_steps = args.nStep,
                 nb_cls = args.nClsEpisode,
                 nb_qry = args.nQuery * args.nClsEpisode,
                 lr_dni = args.lr_dni,
                 dni_hidden_size = args.dniHiddenSize)

netFeat = netFeat.to(device)
netSSR = netSSR.to(device)


args.ckptPth = os.path.join(args.ckptPth, 'outputs/netSSRBest.pth')
param = torch.load(args.ckptPth)
netFeat.load_state_dict(param['netFeat'])
netSSR.load_state_dict(param['SSR'])
msg = '\nLoading networks from {}'.format(args.ckptPth)
logger.info(msg)

msg = 'Testing model loading from {}...\n \n'.format(args.ckptPth)
logger.info(msg)


netSSR.eval()
netFeat.eval()
acc, ci95 = test(testLoader, netSSR, netFeat, args.nEpisode, device, logger, args.nFeat)




