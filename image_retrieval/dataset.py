import pickle
import numpy as np
from tqdm import tqdm 

def trans_multi(A):
    '''
    compute A @ A.T 
    '''
    rows = A.shape[0]
    result = np.empty((rows, rows), dtype=A.dtype)
    
    nb_loop = rows // 10000
    last_iter = rows - nb_loop * 10000
    for r in tqdm(range(nb_loop)):
        result[r*10000 : (r+1)*10000] = A[r*10000 : (r+1)*10000] @ A.T
    
    result[-last_iter : ] = A[-last_iter : ] @ A.T
        
    return result
   
def get_sfm120k_loader() : 
    '''
    feat['database'] shape : (91642, 2048) 
    feat['cluster'] shape : (91642, ) 
    '''
    with open('./data/landmark/gl18_resnet101_sfm120k.pkl', 'rb') as f :
        feat = pickle.load(f)
    
    raw_feat = feat['database'].astype(np.float32).T
    cluster = np.array(feat['cluster']).astype(np.int32)
    score = trans_multi(raw_feat)
    
    return raw_feat, cluster, score

def get_test_loader() : 

    with open('./data/landmark/gl18_resnet101_roxford5k.pkl', 'rb') as f :
        info_oxford = pickle.load(f)
        
    with open('./data/landmark/gl18_resnet101_rparis6k.pkl', 'rb') as f :
        info_paris = pickle.load(f)
                         
    return info_oxford, info_paris  
    
def mAP_sample(binary_label) : 
    TP_index = np.where(binary_label)[0]
    mAP = np.sum(np.arange(1, len(TP_index) + 1) / (TP_index + 1)) / (len(TP_index) + 1e-7)
    return mAP

def get_mAP_sequence(qry_idx, db_idx, score, cluster_gt, nb_neigh) : 
    rank = np.argsort(-1 * score[qry_idx, db_idx])
    binary_label = cluster_gt[db_idx] == cluster_gt[qry_idx]
    return mAP_sample(binary_label[rank[:nb_neigh]]), np.sum(binary_label[rank[:nb_neigh]])
    

def sample_train_sequence(nb_qry, score, cluster, nb_neigh, max_mAP = 0.8, min_true_pos = 5) :
    
    nb_sample = score.shape[0]
    sequence_size = 1000
    sequence_idx = np.zeros((nb_qry, sequence_size + 1), dtype=np.int32)
    count = 0
    
    while count < nb_qry : 
    
        sample_idx = np.random.choice(nb_sample, sequence_size + 1, replace=False)
        mAP, nb_true_pos = get_mAP_sequence(sample_idx[0], sample_idx[1:], score, cluster, nb_neigh)
        if mAP < max_mAP and nb_true_pos > min_true_pos: 
            
            sequence_idx[count, 0] = sample_idx[0]
            sequence_idx[count, 1:] = sample_idx[1:]
            count += 1
            
    return sequence_idx

if __name__ == '__main__' :         

    feat, cluster, score= get_sfm120k_loader()                                                           
    nb_qry = 1000
    max_mAP = 0.8
    min_true_pos = 5
    
    sequence_idx, mAP_avg = sample_train_sequence(nb_qry, score, cluster, max_mAP, min_true_pos)
    print (mAP_avg.mean())


