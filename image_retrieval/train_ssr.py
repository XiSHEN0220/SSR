import sys 
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn


import json
import os
import argparse
import numpy as np
import itertools
import time
import glob

import utils 
import ssr
import dataset 
import eval_landmark

    
def InfoBCE(similarity_refine, topk_label, scale) : 
    pos = (similarity_refine[topk_label == 1] * scale).exp()
    sum_neg = (similarity_refine[topk_label == 0] * scale).exp().sum()
    loss = (- (pos / (sum_neg + pos)).log()).mean()
    return loss
    

# Training
def train(nbIter, raw_feat, cluster, score, net_ssr, optimizer, nb_neigh, evalIter, maxMAP, minTruePos, logger):
    
    net_ssr.train()

    losses = utils.AverageMeter()
    sequence_idx = dataset.sample_train_sequence(evalIter, score, cluster, nb_neigh, maxMAP, minTruePos)
    refine_mAP = np.zeros(evalIter)
    org_mAP = np.zeros(evalIter)
    
    
    for sample_idx in range(evalIter) :

        query_idx = sequence_idx[sample_idx, 0]
        query_label = cluster[query_idx]
        
        # the 1st feature is the query one
        sample_database = sequence_idx[sample_idx, 1:]
        database_score =  score[query_idx, sample_database]
        raw_rank = np.argsort(-1 * database_score)
        database_idx = sample_database[raw_rank[:nb_neigh]] 
        database_label = cluster[database_idx]
        
        org_mAP[sample_idx] = dataset.mAP_sample(database_label == query_label)
        

        ## query feature
        query_weight = torch.from_numpy(raw_feat[query_idx]).cuda()
        query_feat = query_weight.unsqueeze(0).requires_grad_(True) # 1 * feat dim
        
        ## database feature
        database_feat = torch.from_numpy(raw_feat[database_idx]).cuda() 
        database_feat = database_feat.requires_grad_(True) # k * feat dim
        
        ## binary label
        binary_label_numpy = database_label == query_label
        binary_label = torch.from_numpy(binary_label_numpy.astype(np.float32)).cuda().view(-1)
        
        optimizer.zero_grad()
        
        similarity_refine = net_ssr(query_feat, database_feat)
        loss = InfoBCE(similarity_refine, binary_label, net_ssr.get_scale())
        
        
        
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), 1)
        
        similarity_refine = similarity_refine.cpu().detach().numpy()
        rerank = np.argsort(-similarity_refine)
        refine_mAP[sample_idx] = dataset.mAP_sample(binary_label_numpy[rerank])
        
        if sample_idx % 1000 == 999:
            
            
            msg = '{} iter {:d}, Loss: {:.3f} |  mAP : {:.3f} -- > {:.3f}'.format(time.asctime(time.localtime()), nbIter + sample_idx, losses.avg, org_mAP[:sample_idx].mean(), refine_mAP[:sample_idx].mean())
            print(msg)
        
    return nbIter + evalIter, losses.avg, refine_mAP.mean()
    
# Test
def test(info_oxford, info_paris, net_ssr, nb_neigh, logger): 
    
    '''
    Only test not using as validation
    '''
    net_ssr.eval()
    
    
    ### Oxford
    oxford_db = info_oxford['database'].T
    oxford_query = info_oxford['query'].T
    oxford_gt = info_oxford['gt']

    nb_query = oxford_query.shape[0]
    
    scores = np.dot( oxford_query, oxford_db.T)
    ranks = np.argsort(-scores, axis=1)
    
    rank_refine = np.copy( ranks )
    
    
    for i in range(nb_query) : 
        
        
        database_idx = ranks[i, :nb_neigh]
        
        database = torch.from_numpy(oxford_db[database_idx]).cuda()
        query_weight = torch.from_numpy(oxford_query[i]).cuda()
        query_feat = query_weight.unsqueeze(0).requires_grad_(True)
        
        database_feat = database.requires_grad_(True)
        similarity_refine = net_ssr(query_feat, database_feat)
        similarity_refine = similarity_refine.detach().cpu().numpy()
        rerank = np.argsort(-similarity_refine)
        rank_refine[i, :nb_neigh] = database_idx[rerank]
        
    
    oxford_mapM_org, oxford_mapH_org = eval_landmark.compute_map_and_print('roxford5k', ranks.T, oxford_gt, kappas=[nb_neigh])
    
    logger.info('rOxford, after refine...')
    oxford_mapM_refine, oxford_mapH_refine = eval_landmark.compute_map_and_print('roxford5k', rank_refine.T, oxford_gt, kappas=[nb_neigh])
    
    logger.info('\n')
    ### Paris
    paris_db = info_paris['database'].T
    paris_query = info_paris['query'].T
    paris_gt = info_paris['gt']

    nb_query = paris_query.shape[0]
    
    scores = np.dot( paris_query, paris_db.T)
    ranks = np.argsort(-scores, axis=1)
    
    rank_refine = np.copy( ranks )
    
    
    for i in range(nb_query) : 
        
        
        database_idx = ranks[i, :nb_neigh] # the 1st feature is the query one
        
            
        database = torch.from_numpy(paris_db[database_idx]).cuda()

        query_weight = torch.from_numpy(paris_query[i]).cuda()
        query_feat = query_weight.unsqueeze(0).requires_grad_(True)
        
        database_feat = database.requires_grad_(True)
        database_feat = database_feat.unsqueeze(0)
        
        
        similarity_refine = net_ssr(query_feat, database_feat)
        
        similarity_refine = similarity_refine.detach().cpu().numpy()
        rerank = np.argsort(-similarity_refine)
        
        rank_refine[i, :nb_neigh] = database_idx[rerank]
        
    
    paris_mapM_org, paris_mapH_org =eval_landmark.compute_map_and_print('rparis6k', ranks.T, paris_gt, kappas=[nb_neigh])
    
    logger.info('rParis, after refine...')
    paris_mapM_refine, paris_mapH_refine =eval_landmark.compute_map_and_print('rparis6k', rank_refine.T, paris_gt, kappas=[nb_neigh])
    
    return oxford_mapM_refine, oxford_mapH_refine, paris_mapM_refine, paris_mapH_refine
            
def main(gpu, lr, lr_dni, trainIter, resumeSSR, nStep, momentum, weightDecay, evalIter, nb_neigh, maxMAP, minTruePos) :

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    
    raw_feat, cluster, score = dataset.get_sfm120k_loader()
    
    info_oxford, info_paris = dataset.get_test_loader()
    
    
    net_ssr = ssr.SSR(raw_feat.shape[1], nStep, nb_neigh, lr_dni, dni_hidden_size=1024)
    
    outDir = 'cache/sfm120k_Step{:d}_Rerank{:d}_LrDni{:.5f}_maxMAP{:.1f}_MinTruePos{:d}'.format(nStep, nb_neigh, lr_dni, maxMAP, minTruePos)
    
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
    
    logger = utils.get_logger_logpath(os.path.join(outDir, 'log'))
    
    if nStep > 1 and resumeSSR is None:
        
        step_pre = 'Step{:d}'.format(nStep - 1)
        step = 'Step{:d}'.format(nStep)
        
        resumeSSR = glob.glob( outDir.replace(step, step_pre)+'*'+'/netBestMAP.pth' )[0]
        
        # set lr and iterations for step > 1
        lr = 1e-3
        trainIter = 10000
        evalIter = 1000
        
        
    if resumeSSR : 
        msg = 'Loading pretrained weight from {}'.format(resumeSSR)
        logger.info(msg)
        param = torch.load(resumeSSR)
        net_ssr.load_state_dict(param)
        
    net_ssr.cuda()    
    
    oxford_mapM_refine, oxford_mapH_refine, paris_mapM_refine, paris_mapH_refine = test(info_oxford, info_paris, net_ssr, nb_neigh, logger)
    
    bestMAP = oxford_mapM_refine * 0.25 + oxford_mapH_refine * 0.25 + paris_mapM_refine * 0.25 + paris_mapH_refine * 0.25 
    logger.info('Raw mAP on the test set is : {:.3f}, Oxford M : {:.3f}, H : {:.3f}; Paris M : {:.3f}, H : {:.3f}'.format(bestMAP, oxford_mapM_refine, oxford_mapH_refine, paris_mapM_refine, paris_mapH_refine))
    
    
    
    optimizer = torch.optim.SGD(itertools.chain(*[net_ssr.parameters()]), 
                                                 lr, 
                                                 momentum=momentum, 
                                                 weight_decay=weightDecay)
    
    history = {'BestMAP':[],  'trainLoss':[],  'trainMAP':[], 'testMAP' : [], 'rOxfordM' : [], 'rParisM' : [], 'rOxfordH' : [], 'rParisH' : []}
    
    nIter = 0
    
    
    
    while nIter < trainIter : 
        nIter, trainLoss, trainMAP = train(nIter, raw_feat, cluster, score, net_ssr, optimizer, nb_neigh, evalIter, maxMAP, minTruePos, logger)
        
        
        oxford_mapM_refine, oxford_mapH_refine, paris_mapM_refine, paris_mapH_refine = test(info_oxford, info_paris, net_ssr, nb_neigh, logger)
    
        testMAP = oxford_mapM_refine * 0.25 + oxford_mapH_refine * 0.25 + paris_mapM_refine * 0.25 + paris_mapH_refine * 0.25 
        logger.info('{:d} iter, Test mAP on the test set is : {:.3f} (Best {:.3f}), Oxford M : {:.3f}, H : {:.3f}; Paris M : {:.3f}, H : {:.3f}'.format(nIter, testMAP, bestMAP, oxford_mapM_refine, oxford_mapH_refine, paris_mapM_refine, paris_mapH_refine))
        

        history['BestMAP'].append( bestMAP )
        history['trainLoss'].append( trainLoss )
        history['trainMAP'].append( trainMAP )
        
        history['testMAP'].append( testMAP )
        
        history['rOxfordM'].append( oxford_mapM_refine )
        history['rOxfordH'].append( oxford_mapH_refine )
        history['rParisM'].append( paris_mapM_refine )
        history['rParisH'].append( paris_mapH_refine )
        
        with open(os.path.join(outDir, 'history.json'), 'w') as f : 
            json.dump(history, f)
        
        if testMAP > bestMAP : 
            logger.info('\nSaving Best...')
            logger.info('\tmAP improved from {:.3f} to {:.3f}!!!'.format(bestMAP, testMAP))

            torch.save(net_ssr.state_dict(), os.path.join(outDir, 'netBestMAP.pth'))

            bestMAP = testMAP 

    msg = 'mv {} {}'.format(outDir, '{}_mAP_{:.3f}'.format(outDir, bestMAP))
    logger.info(msg)
    os.system(msg)
    
    


if __name__ == '__main__' : 
                        
    parser = argparse.ArgumentParser(description='SSR on landmark retrieval task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    
    parser.add_argument('--lr-dni', type = float, default=1e-3, help='learning rate for dni')
    
    parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    
    
    parser.add_argument('--resumeSSR', type=str, help='resume path directory for the SSR module')

    
    parser.add_argument('--trainIter', type = int, default = 60000, help='nb of iterations')
    parser.add_argument('--gpu', type = str, default='0', help='gpu devices')
    parser.add_argument('--evalIter', type = int, default=1000, help='nb of training iteration to do one validation')
    
    parser.add_argument('--nStep', type = int, default=1, help='nb of update')

    parser.add_argument('--nNeigh', type = int, default=100, help='nb of neighbours')

    parser.add_argument('--maxMAP', type = float, default=0.8, help='sample sequence that mAP smaller than this value')
    
    parser.add_argument('--minTruePos', type = int, default=5, help='sample sequence that true positive items are larger than this value')
    
    
    args = parser.parse_args()
    print (args)
    
    main(args.gpu, args.lr, args.lr_dni, args.trainIter, args.resumeSSR, args.nStep, args.momentum, args.weightDecay, args.evalIter, args.nNeigh, args.maxMAP, args.minTruePos)

