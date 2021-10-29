import torch
import torch.nn as nn

import os
import argparse

import dataset
import ssr
import numpy as np
import eval_landmark

import utils
import sklearn.metrics as metrics
import k_reciprocal

def compute_dist(query, db) : 

    query_db_dist = metrics.pairwise.pairwise_distances(query, db, n_jobs = -1)
    query_query_dist = metrics.pairwise.pairwise_distances(query, query, n_jobs = -1)
    db_db_dist = metrics.pairwise.pairwise_distances(db, db, n_jobs = -1)
    
    return query_db_dist, query_query_dist, db_db_dist



def test_info(info, k1, k2, lambda_value, dataset):
    
    '''
    Test k-reciprocal on one landmark dataset
    '''
    
    ### Oxford
    
    db = info['database'].T
    query = info['query'].T
    gt = info['gt']

    nb_query = query.shape[0]
    
    query_db_dist, query_query_dist, db_db_dist = compute_dist(query, db)
    
    query_db_score = k_reciprocal.re_ranking_list(query_db_dist, query_query_dist, db_db_dist, k1, k2, lambda_value)
    rank = np.argsort(query_db_score, axis=1)
    
    mapM_krcpk, mapH_krcpk = eval_landmark.compute_map_and_print(dataset, rank.T, gt, kappas=[100])
    
    return mapM_krcpk, mapH_krcpk, rank
    
# Test
def test(info_oxford, info_paris, k1, k2, lambda_value, nb_neigh, logger, net_ssr): 
    
    '''
    Only test not using as validation
    '''
    
    ### Oxford
        
    mapM_krcpk_oxford, mapH_krcpk_oxford, rank_oxford = test_info(info_oxford, k1, k2, lambda_value, 'roxford5k')
     
    net_ssr.eval()

    oxford_db = info_oxford['database'].T
    oxford_query = info_oxford['query'].T
    oxford_gt = info_oxford['gt']

    nb_query = oxford_query.shape[0]
    rank_refine = np.copy( rank_oxford )

    for i in range(nb_query) : 


        database_idx = rank_oxford[i, :nb_neigh]

        database = torch.from_numpy(oxford_db[database_idx]).cuda()
        query_weight = torch.from_numpy(oxford_query[i]).cuda()
        query_feat = query_weight.unsqueeze(0).requires_grad_(True)

        database_feat = database.requires_grad_(True)
        similarity_refine = net_ssr(query_feat, database_feat)
        similarity_refine = similarity_refine.detach().cpu().numpy()
        rerank = np.argsort(-similarity_refine)
        rank_refine[i, :nb_neigh] = database_idx[rerank]


    oxford_mapM_refine, oxford_mapH_refine = eval_landmark.compute_map_and_print('roxford5k', rank_refine.T, oxford_gt, kappas=[nb_neigh])
        
    ### Paris
    mapM_krcpk_paris, mapH_krcpk_paris, rank_paris = test_info(info_paris, k1, k2, lambda_value, 'rparis6k')
    
    paris_db = info_paris['database'].T
    paris_query = info_paris['query'].T
    paris_gt = info_paris['gt']

    nb_query = paris_query.shape[0]
    rank_refine = np.copy( rank_paris )


    for i in range(nb_query) : 


        database_idx = rank_paris[i, :nb_neigh] # the 1st feature is the query one
        database = torch.from_numpy(paris_db[database_idx]).cuda()

        query_weight = torch.from_numpy(paris_query[i]).cuda()
        query_feat = query_weight.unsqueeze(0).requires_grad_(True)

        database_feat = database.requires_grad_(True)
        database_feat = database_feat.unsqueeze(0)


        similarity_refine = net_ssr(query_feat, database_feat)

        similarity_refine = similarity_refine.detach().cpu().numpy()
        rerank = np.argsort(-similarity_refine)

        rank_refine[i, :nb_neigh] = database_idx[rerank]


    paris_mapM_refine, paris_mapH_refine = eval_landmark.compute_map_and_print('rparis6k', rank_refine.T, paris_gt, kappas=[nb_neigh])
    
    msg = 'K-Reciprocal --> rOxford M : {:.3f}, rOxford H : {:.3f}; rParis M : {:.3f}, rParis H : {:.3f}; Avg : {:.3f}'.format(mapM_krcpk_oxford,
               mapH_krcpk_oxford,
               mapM_krcpk_paris,
               mapH_krcpk_paris,
               mapM_krcpk_oxford / 4 + mapH_krcpk_oxford / 4 + mapM_krcpk_paris / 4 + mapH_krcpk_paris / 4)
    logger.info(msg)
    
    msg = '\t K-Reciprocal + SSR --> rOxford M : {:.3f}, rOxford H : {:.3f}; rParis M : {:.3f}, rParis H : {:.3f}; Avg : {:.3f}'.format(oxford_mapM_refine,
                             oxford_mapH_refine,
                             paris_mapM_refine,
                             paris_mapH_refine,
                             oxford_mapM_refine/4 + oxford_mapH_refine/4 + paris_mapM_refine/4 + paris_mapH_refine/4)
    logger.info(msg)
    
    
def main(gpu, SSR_dir, k1, k2, lambda_value) :

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    keys = SSR_dir.split('_')
    step = [int(key.split('Step')[1]) for key in keys if 'Step' in key][0]
    lr_dni = [float(key.split('LrDni')[1]) for key in keys if 'LrDni' in key][0]
    nb_neigh = [int(key.split('Rerank')[1]) for key in keys if 'Rerank' in key][0]
    
    logger = utils.get_logger_logpath(os.path.join(SSR_dir, 'Krcpc.txt'))
    
    logger.info('K1 : {:d}, K2 : {:d}, Lambda : {:.3f}, Step : {:d}, Lr-Dni : {:.5f}, Nb Neigh : {:d} ...'.format(k1, k2, lambda_value, step, lr_dni, nb_neigh))
    
    info_oxford, info_paris = dataset.get_test_loader()
    net_ssr = ssr.SSR(2048, step, nb_neigh, lr_dni, dni_hidden_size=1024)
        
    SSR_pth = os.path.join(SSR_dir, 'netBestMAP.pth')
    msg = 'Loading pretrained weight from {}'.format(SSR_pth)
    logger.info(msg)
    param = torch.load(SSR_pth)
    net_ssr.load_state_dict(param)

    net_ssr.cuda()     
    
    test(info_oxford, info_paris, k1, k2, lambda_value, nb_neigh, logger, net_ssr)
    

if __name__ == '__main__' : 
                        
    parser = argparse.ArgumentParser(description='Train classification but eval on landmark retrieval task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument('--SSR-dir', type=str, help='resume path directory for the NUS module, default None')

    parser.add_argument('--gpu', type = str, default='0', help='gpu devices, default 0')
    
    parser.add_argument( '--k1', default=20, type=int, help='k1, default is 20 used in the org paper')

    parser.add_argument( '--k2', default=6, type=int, help='k2, default is 6 used in the org paper')

    parser.add_argument( '--lambda-value', default=0.3, type=float, help='lambda, default is 0.3 used in the org paper')

    
    
    args = parser.parse_args()
    print (args)
    main(args.gpu, args.SSR_dir, args.k1, args.k2, args.lambda_value)

