import torch
import torch.nn as nn

import os
import argparse

import dataset
import ssr
import numpy as np
import eval_landmark

import utils

from sklearn.preprocessing import normalize
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def dual_solve(X, y):
    #Initializing values and computing H. Note the 1. to force to float type
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.

    #Converting into cvxopt format
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters (change default to decrease tolerance)
    cvxopt_solvers.options['show_progress'] = False
    cvxopt_solvers.options['abstol'] = 1e-10
    cvxopt_solvers.options['reltol'] = 1e-10
    cvxopt_solvers.options['feastol'] = 1e-10

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def DQE(K, X):
    nb_neg = X.shape[0] - K
    y = np.array([1.] * K + [-1.] * nb_neg)
    alphas = dual_solve(X, y)
    w = alphas.flatten() * y
    sum_feat = np.sum(X * w.reshape(-1, 1), axis=0)
    return normalize(sum_feat[:, np.newaxis], axis=0).ravel()


def AQE(qry_feat, topk_feat):
    all_feat = np.concatenate((qry_feat.reshape(1, -1), topk_feat), axis=0)
    sum_feat = np.sum(all_feat, axis=0)
    return normalize(sum_feat[:, np.newaxis], axis=0).ravel()


def AQEwD(qry_feat, topk_feat):
    all_feat = np.concatenate((qry_feat.reshape(1, -1), topk_feat), axis=0)
    K = all_feat.shape[0]
    w = ((K - np.arange(K)) / K).reshape(-1, 1)
    sum_feat = np.sum(all_feat * w, axis=0)
    return normalize(sum_feat[:, np.newaxis], axis=0).ravel()


def alphaQE(qry_feat, topk_feat, alpha=3):
    all_feat = np.concatenate((qry_feat.reshape(1, -1), topk_feat), axis=0)
    w = (np.dot(qry_feat, all_feat.T) ** alpha).reshape(-1, 1)
    sum_feat = np.sum(all_feat * w, axis=0)
    return normalize(sum_feat[:, np.newaxis], axis=0).ravel()


def update_info_QE(info, QE_method, QE_neigh, dataset='roxford5k', isDQE = False) : 
    
    '''
    QE on one landmark dataset, which gives up the updated query
    '''
    
    
    db = info['database'].T
    query = info['query'].T
    gt = info['gt']

    nb_query = query.shape[0]
    
    scores = np.dot( query, db.T)
    ranks = np.argsort(-scores, axis=1)
    
    update_qry = np.zeros(query.shape, dtype=np.float32)
    
    for i in range(nb_query) : 
        
        retrieval_idx = ranks[i, :QE_neigh] 
        if isDQE : 
            neg_feat = db[ranks[i, -(QE_neigh+1):]]
            X = np.concatenate((query[i].reshape(1, -1), db[retrieval_idx].reshape(QE_neigh, -1), neg_feat.reshape(QE_neigh+1, -1)), axis=0)
            expand_qry_feat = DQE(QE_neigh + 1, X)
        else : 
            expand_qry_feat = QE_method(qry_feat = query[i], topk_feat = db[retrieval_idx])
        update_qry[i] = expand_qry_feat
    
    scores = np.dot( update_qry, db.T)
    rank_refine = np.argsort(-scores, axis=1)
    mapM_refine, mapH_refine = eval_landmark.compute_map_and_print(dataset, rank_refine.T, gt, kappas=[100])
    
    info['query'] = update_qry.T
    return info, mapM_refine, mapH_refine

def test(info_oxford, info_paris, net_ssr, nb_neigh, QE_method, QE_neigh, isDQE, logger, QE_name) : 
    
    '''
    Test QE + SSR on one landmark dataset
    '''
    info_oxford, mapM_QE_oxford, mapH_QE_oxford = update_info_QE(info_oxford, QE_method, QE_neigh, 'roxford5k', isDQE)
    info_paris, mapM_QE_paris, mapH_QE_paris = update_info_QE(info_paris, QE_method, QE_neigh, 'rparis6k', isDQE)
    
    net_ssr.eval()

    oxford_db = info_oxford['database'].T
    oxford_query = info_oxford['query'].T
    oxford_gt = info_oxford['gt']

    nb_query = oxford_query.shape[0]
    scores = np.dot( oxford_query, oxford_db.T)
    rank_oxford = np.argsort(-scores, axis=1)
    
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
    paris_db = info_paris['database'].T
    paris_query = info_paris['query'].T
    paris_gt = info_paris['gt']

    nb_query = paris_query.shape[0]
    scores = np.dot( paris_query, paris_db.T)
    rank_paris = np.argsort(-scores, axis=1)
    
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
    
    msg = '{} --> rOxford M : {:.3f}, rOxford H : {:.3f}; rParis M : {:.3f}, rParis H : {:.3f}; Avg : {:.3f}'.format(QE_name,
               mapM_QE_oxford,
               mapH_QE_oxford,
               mapM_QE_paris,
               mapH_QE_paris,
               mapM_QE_oxford / 4 + mapH_QE_oxford / 4 + mapM_QE_paris / 4 + mapH_QE_paris / 4)
    logger.info(msg)
    
    msg = '\t {} + SSR  --> rOxford M : {:.3f}, rOxford H : {:.3f}; rParis M : {:.3f}, rParis H : {:.3f}; Avg : {:.3f}'.format(QE_name,
                 oxford_mapM_refine,
                 oxford_mapH_refine,
                 paris_mapM_refine,
                 paris_mapH_refine,
                 oxford_mapM_refine/4 + oxford_mapH_refine/4 + paris_mapM_refine/4 + paris_mapH_refine/4)
    logger.info(msg)
    
    
    

    
def main(gpu, SSR_dir, QE_name, QE_neigh) :

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    keys = SSR_dir.split('_')
    step = [int(key.split('Step')[1]) for key in keys if 'Step' in key][0]
    lr_dni = [float(key.split('LrDni')[1]) for key in keys if 'LrDni' in key][0]
    nb_neigh = [int(key.split('Rerank')[1]) for key in keys if 'Rerank' in key][0]
    
    logger = utils.get_logger_logpath(os.path.join(SSR_dir, 'QE.txt'))
    
    logger.info('{}, QE {:d} neighs, Step : {:d}, Lr-Dni : {:.5f}, Nb Neigh : {:d} ...'.format(QE_name, QE_neigh, step, lr_dni, nb_neigh))
    
    info_oxford, info_paris = dataset.get_test_loader()
    net_ssr = ssr.SSR(2048, step, nb_neigh, lr_dni, dni_hidden_size=1024)
        
    SSR_pth = os.path.join(SSR_dir, 'netBestMAP.pth')
    msg = 'Loading pretrained weight from {}'.format(SSR_pth)
    logger.info(msg)
    param = torch.load(SSR_pth)
    net_ssr.load_state_dict(param)

    net_ssr.cuda()     
    
    isDQE = False 
    if QE_name == 'AQE':
        QE_method = AQE
        
    elif QE_name == 'AQEwD':
        QE_method = AQEwD
    
    elif QE_name == 'alphaQE':
        QE_method = alphaQE
    
    elif QE_name == 'DQE' :
        QE_method = DQE
        isDQE = True
    
    test(info_oxford, info_paris, net_ssr, nb_neigh, QE_method, QE_neigh, isDQE, logger, QE_name)
    

if __name__ == '__main__' : 
                        
    parser = argparse.ArgumentParser(description='Test SSR with / without QE method', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument('--SSR-dir', type=str, help='resume path directory for the NUS module, default None')

    parser.add_argument('--gpu', type = str, default='0', help='gpu devices, default 0')
    
    parser.add_argument('--QE-name', type=str, choices=['AQE', 'AQEwD', 'alphaQE', 'DQE'], help='query expansion name')
    parser.add_argument( '--QE-neigh', default=1, type=int, help='QE neigh')
    
    args = parser.parse_args()
    print (args)
    
    main(args.gpu, args.SSR_dir, args.QE_name, args.QE_neigh)

