
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
import k_reciprocal # k-reciprocal



import argparse
import sys 
import torch.nn.functional as F
import numpy as np


              
def test(dataloader, netFeat, nEpisode, device, logger, nFeat, nShot, k1_list, k2_list, lambda_list, episodeAccLog) :

    nEpisode = nEpisode
    logger.info('\n\nTest mode: randomly sample {:d} episodes of {:d}-shot...'.format(nEpisode, nShot))
    
    for batchIdx in range(nEpisode):
        data = dataloader.getEpisode() 
        data = utils.to_device(data, device)

        SupportTensor = data['SupportTensor']
        QueryTensor = data['QueryTensor']
        QueryLabel = data['QueryLabel'].cpu().numpy()
        
        
        with torch.no_grad() :
            clsdim, nb_supp = SupportTensor.size()[0], SupportTensor.size()[1]
            nb_query = QueryTensor.size()[0]
            SupportFeat = netFeat(SupportTensor.contiguous().view(-1, 3, inputW, inputH))
            SupportFeat = SupportFeat.view(clsdim, nb_supp, nFeat)
            SupportFeat = SupportFeat.mean(1)
            QueryFeat = netFeat(QueryTensor.contiguous().view(-1, 3, inputW, inputH))
            SupportFeat, QueryFeat = F.normalize(SupportFeat, dim=1), F.normalize(QueryFeat, dim=1)
            
            query_db_dist = utils.pairwise_cuda(QueryFeat, SupportFeat) 
            query_query_dist = utils.pairwise_cuda(QueryFeat, QueryFeat)
            db_db_dist = utils.pairwise_cuda(SupportFeat, SupportFeat)
            
            
        for k1 in k1_list : 
            for k2 in k2_list : 
                for lambda_value in lambda_list :     
                    clsScore = k_reciprocal.re_ranking_list(query_db_dist, query_query_dist, db_db_dist, k1, k2, lambda_value)
                    pred = np.argmin(clsScore, axis=1)
                    acc1 = np.sum(pred == QueryLabel) / float(pred.shape[0]) * 100
                    episodeAccLog[(k1, k2, lambda_value)].append(acc1)
        
    best_param = 0
    best_perf = (0, 0)
    
    for k1 in k1_list : 
        for k2 in k2_list : 
            for lambda_value in lambda_list : 
                mean, ci95 = utils.getCi(episodeAccLog[(k1, k2, lambda_value)])
                msg = '\t\t\t-->\t\tK1 = {:d}, K2 = {:d}, Lambda = {:.3f}; Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(k1, k2, lambda_value, mean, ci95)
                logger.info(msg)
                
                if mean > best_perf[0]: 
                    best_param = (k1, k2, lambda_value)
                    best_perf = (mean, ci95)
                    
    msg = '\n\t\t\tBest param is K1 = {:d}, K2 = {:d}, Lambda = {:.3f} \t Best perf is  acc {:.3f}%, ci95 {:.3f}%'.format(best_param[0], best_param[1], best_param[2], best_perf[0], best_perf[1])

    logger.info(msg)
    
        

        

#############################################################################################
## Parameters
parser = argparse.ArgumentParser(description='Transductive Few-shot Evaluation with K-Recoprocal')

parser.add_argument('--nEpisode', type = int, default = 2000, help='nb episode')
parser.add_argument('--gpu', type=int, help='which gpu?')
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'Cifar', 'tieredImageNet'], help='Which dataset? Should modify normalization parameter')
parser.add_argument('--resumeFeatPth', type = str, help='resume feature path')    

# Validation
parser.add_argument('--nSupport', type = int, default=1, help='support set')
parser.add_argument('--nQuery', type = int, default=15, help='query set')
parser.add_argument('--nClsEpisode', type = int, default=5, help='nb of classes in each set')

parser.add_argument('--architecture', type = str, default='WRN_28_10', choices=['WRN_28_10', 'ConvNet_4_64', 'ResNet12'], help='arch')

parser.add_argument( '--k1', nargs='+', default=[10, 20, 30], type=int, help='k1 list, default is [20] used in the org paper')

parser.add_argument( '--k2', nargs='+', default=[5, 10, 15], type=int, help='k2 list, default is [6] used in the org paper')

parser.add_argument( '--lambda-list', nargs='+', default=[0.2, 0.5, 0.8], type=float, help='lambda list, default is [0.3] used in the org paper')


args = parser.parse_args()
print (args)

# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda')

logPath = args.resumeFeatPth.replace('.pth', '.test_no_training')
logger = utils.get_logger_logpath(logPath)

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

## loading feature net from the base classifier
if args.resumeFeatPth : 
    param = torch.load(args.resumeFeatPth)
    netFeat.load_state_dict(param)
    msg = '\nLoading networks from {}'.format(args.resumeFeatPth)
    logger.info(msg)


netFeat.eval()
netFeat = netFeat.to(device)

episodeAccLog = {}
for k1 in args.k1 : 
        for k2 in args.k2 : 
            for lambda_value in args.lambda_list : 
                episodeAccLog[(k1, k2, lambda_value)] = []
                
test(testLoader, netFeat, args.nEpisode, device, logger, args.nFeat, args.nSupport, args.k1, args.k2, args.lambda_list, episodeAccLog)
    
