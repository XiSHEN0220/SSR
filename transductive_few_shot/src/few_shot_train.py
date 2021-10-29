import torch
import torch.nn as nn
import random
import itertools
import json
import os

from glob import glob

import networks # get_networks
import ssr # SSR
import dataset # get datase setting 
import dataloader # train / val dataloader 
import utils # other functions





def train(trainLoader, netSSR, netFeat, iter_epoch, criterion, optimizer, lr_schedule, nFeat, device, inputW, inputH) :
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    
    for episode in range(iter_epoch):
        data = trainLoader.getEpisode()
        data = utils.to_device(data, device)
        
        with torch.no_grad() :
            SupportTensor, QueryTensor, QueryLabel = data['SupportTensor'], data['QueryTensor'], data['QueryLabel']
            nb_cls, nb_supp = SupportTensor.size()[0], SupportTensor.size()[1]
            nb_query = QueryTensor.size()[0]
            
            SupportFeat = netFeat(SupportTensor.contiguous().view(-1, 3, inputW, inputH))
            QueryFeat = netFeat(QueryTensor.contiguous().view(-1, 3, inputW, inputH))
            SupportFeat, QueryFeat = SupportFeat.contiguous().view(nb_cls, nb_supp, nFeat), QueryFeat.view(nb_query, nFeat)

        
        SupportFeat = SupportFeat.requires_grad_(True)
        QueryFeat = QueryFeat.requires_grad_(True)
        
        optimizer.zero_grad()
        
        clsScore = netSSR(SupportFeat, QueryFeat)
        QueryLabel = QueryLabel.view(-1)

        loss = criterion(clsScore, QueryLabel)

        loss.backward()
        optimizer.step()
        lr_schedule.step()
        acc1 = utils.accuracy(clsScore, QueryLabel, topk=(1, ))
        top1.update(acc1[0].item(), clsScore.size()[0])
        losses.update(loss.item(), QueryFeat.size()[1])
        
        if episode % 500 == 499 : 
            msg = 'Training Episode {:d}, Loss: {:.3f} | Top1: {:.3f}% '.format(episode, losses.avg, top1.avg)
            print (msg)
        

    return losses.avg, top1.avg
        
def val(dataloader, netSSR, netFeat, device, logger, nFeat, inputW, inputH,) :

    nEpisode = len(dataloader)
    logger.info('\n\nValidation mode: pre-defined {:d} episodes...'.format(nEpisode))
    dataloader = iter(dataloader)
            
    episodeAccLog = []
    top1 = utils.AverageMeter()

    for batchIdx in range(nEpisode):
        data = next(dataloader)
        data = utils.to_device(data, device)
        
        SupportTensor = data['SupportTensor'][0]   
        QueryTensor = data['QueryTensor'][0]
        QueryLabel = data['QueryLabel'][0]
         

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
         
        if batchIdx % 500 == 499 : 
            msg = 'Validation Episode {:d}, Top1: {:.3f}% '.format(batchIdx, top1.avg)
            print (msg)
        episodeAccLog.append(acc1[0].item())
        

    mean, ci95 = utils.getCi(episodeAccLog)
    logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))

    return mean, ci95        

def main(args) :         
    #############################################################################################
    
    args.seed = 300
    # logging to the file and stdout
    
    cacheDir = os.path.join("cache", '{}_Step{}_LrDni{:.5f}_Hidden{}'.format(args.expName, args.nStep, args.lr_dni, args.dniHiddenSize))
    
    if args.nStep > 1:
        args.lr = 0.01
        args.ckptPth = glob(cacheDir.replace(f'Step{args.nStep}', f'Step{args.nStep-1}')+'*'+'/outputs/netSSRBest.pth')[0]
        print(f'ckpt path of step {args.nStep-1} ----> ', args.ckptPth)
        args.nbIter = 10000
        args.schedule = [10000]
    
    
    logDir = os.path.join(cacheDir, 'logs')
    outDir = os.path.join(cacheDir, 'outputs')
    if not os.path.exists(cacheDir):
        os.mkdir(cacheDir)
    if not os.path.exists(logDir):
        os.mkdir(logDir)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    logger = utils.get_logger(logDir, args.expName)
        
    # fix random seed to reproduce results
    utils.set_random_seed(args.seed)
    logger.info('Start experiment with random seed: {:d}'.format(args.seed))
    logger.info(args)

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.gpu != '':
        args.cuda = True
    device = torch.device('cuda' if args.cuda else 'cpu')

    #############################################################################################
    ## datasets setting 
    trainTransform, valTransform, inputW, inputH, \
            trainDir, valDir, testDir, episodeJson, nbCls = \
            dataset.dataset_setting(args.dataset, args.nSupport, args.nQuery)

    ## train / val / test dataloader
    trainLoader = dataloader.EpisodeSampler(imgDir = trainDir,
                                           nClsEpisode = args.nClsEpisode,
                                           nSupport = args.nSupport, 
                                           nQuery = args.nQuery,
                                           transform = trainTransform,
                                           useGPU = args.cuda,
                                           inputW = inputW,
                                           inputH = inputH)

    
                                    
    
    valLoader = dataloader.ValLoaderEpisode(episodeJson,
                                  valDir,
                                  inputW,
                                  inputH,
                                  valTransform,
                                  args.cuda)


    #############################################################################################
    ## Networks
    # backbone 
    netFeat, args.nFeat = networks.get_featnet(args.architecture, inputW, inputH, args.dataset)
    # SSR
    netSSR = ssr.SSR(feat_dim = args.nFeat,
                     q_steps = args.nStep,
                     nb_cls = args.nClsEpisode,
                     nb_qry = args.nQuery * args.nClsEpisode,
                     lr_dni = args.lr_dni,
                     dni_hidden_size = args.dniHiddenSize)
    
    netFeat = netFeat.to(device)
    netSSR = netSSR.to(device)

    ## Optimizer (Only SSR)
    optimizer = torch.optim.SGD(itertools.chain(*[netSSR.parameters(),]),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weightDecay,
                                nesterov=True)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=0.1)

    ## Loss
    criterion = nn.CrossEntropyLoss()

    if args.resumeFeatPth :
        if args.cuda:
            param = torch.load(args.resumeFeatPth)
        else:
            param = torch.load(args.resumeFeatPth, map_location='cpu')
        netFeat.load_state_dict(param)
        msg = '\nLoading netFeat from {}'.format(args.resumeFeatPth)
        logger.info(msg)
        
    if args.ckptPth:
        param = torch.load(args.ckptPth)
        netFeat.load_state_dict(param['netFeat'])
        netSSR.load_state_dict(param['SSR'])
        msg = '\nLoading networks from {}'.format(args.ckptPth)
        logger.info(msg)

    history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : [], 'valBest':[]}

    nEpoch = args.nbIter // args.iter_epoch
    bestAcc = 0.

    for i in range(nEpoch) : 

        netSSR.train()
        netFeat.eval()
        
        loss, top1 = train(trainLoader, netSSR, netFeat, args.iter_epoch, criterion, optimizer, lr_schedule, args.nFeat, device, inputW, inputH)
        
        netSSR.eval()
        netFeat.eval()
        
        acc, ci95 = val(valLoader, netSSR, netFeat, device, logger, args.nFeat, inputW, inputH)

        msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format((i+1) * args.iter_epoch, loss, top1, acc)
        logger.info(msg)
        
        history['trainLoss'].append(loss)
        history['trainAcc'].append(top1)
        history['valAcc'].append(acc)
        history['valBest'].append(max(bestAcc, acc))
                    
        if acc > bestAcc : 
            msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
            logger.info(msg)

            bestAcc = acc
            logger.info('Saving Best')
            torch.save({'netFeat': netFeat.state_dict(),
                        'SSR': netSSR.state_dict(),
                        'nbStep': args.nStep,
                        }, os.path.join(outDir, 'netSSRBest.pth'))
                        
        
    ## Finish training!!!
    outDirUpdate = '{}_{:.3f}'.format(cacheDir, bestAcc)
    msg = 'mv {} {}'.format(cacheDir, outDirUpdate)
    logger.info(msg)
    os.system(msg)

    with open(os.path.join(outDirUpdate, 'history.json'), 'w') as f :
        json.dump(history, f)

    
    
def get_args():
    import argparse
    import sys
    cmdline_args = sys.argv[1:] if sys.argv[0] == __file__ else []
    
    argparser = argparse.ArgumentParser(description='Few shot classification')
    
    argparser.add_argument('--nStep', type=int, default=1, help='nb of step')
    
    argparser.add_argument('--gpu', type=int, default=0, help='which gpu ?')
    
    ## episode    
    argparser.add_argument('--nClsEpisode', type=int, default=5, help='nb cls in each episode')
    
    argparser.add_argument('--nSupport', type=int, default=1, help='nb support in each episode')
    
    argparser.add_argument('--nQuery', type=int, default=15, help='nb queries in each episode')
    
    ## dataset 
    argparser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'Cifar', 'tieredImageNet'], help='Which dataset? Should modify normalization parameter')
    
    ## learning parameters
    argparser.add_argument('--nbIter', type=int, default=30000, help='nb iterations')
    
    argparser.add_argument('--architecture', type=str, default='WRN_28_10', choices = ['WRN_28_10', 'ConvNet_4_64', 'ResNet12'], help='which arch?')
    
    
    argparser.add_argument('--iter-epoch', type=int, default=1000, help='each epoch contains how many iterations? frequence to do validation')
    
    argparser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    
    argparser.add_argument('--lr-dni', type=float, default=1e-3, help='learning rate for dni')
    
    argparser.add_argument('--schedule', type=int, default=[5000], nargs='+', help='learning rate schedule')
    
    argparser.add_argument('--weightDecay', type=float, default=5e-4, help='weight decay')
    
    argparser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    argparser.add_argument('--expName', type=str, default='MiniImageNet_WRN', help='will define the name of output')
    
    argparser.add_argument('--resumeFeatPth', type=str, default='../ckpts/MiniImageNet/WRN_Cos_netFeatBest64.653.pth', help='resumePth')
    
    argparser.add_argument('--ckptPth', type=str, default=None, help='ckptPth')
    
    argparser.add_argument('--dniHiddenSize', type=int, default=4096, help='dni hidden size')
    
    args = argparser.parse_args(args=cmdline_args)
    return args
    
if __name__ == '__main__' : 
                        
    args = get_args()
    
    main(args)

    
    
