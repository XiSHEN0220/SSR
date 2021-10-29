# Code adapted from SIB: https://github.com/hushell/sib_meta_learn/blob/master/sib.py
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
class SSR(nn.Module):
    """
    feat_dim: feature dimension at the input of classifier
    q_steps: number of update used in weights refinement
    topk: topk retrieved images
    lr_dni : learning rate for dni, 1e-3 is used in SIB
    dni_hidden_size: hidden dimension in dni
    """
    def __init__(self, feat_dim, q_steps, k, lr_dni = 1e-3, dni_hidden_size = 2048):
        super(SSR, self).__init__()

        self.feat_dim = feat_dim
        self.q_steps = q_steps
        
        self.k = k ## top-k retrieved images
        self.lr_dni = lr_dni
        
        # scale of classifier
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        
        # dni grad_net
        dni_input_dim = (k + 2) * (1 + k) // 2  - k * (k - 1) // 2
        tmp = torch.ones((k + 2), (k + 2)) ## full adjacency matrix
        tmp[2 : , 2 : ] = 0
        tmp = torch.triu(tmp, diagonal=1)
        tmp = 1 - tmp
        
        self.triu_index_input = (tmp == 0).cuda()
        
        
        ## we employ 2 dni for support and query set 
        self.dni_query = dni_linear(dni_input_dim, dni_input_dim, dni_hidden_size=dni_hidden_size)
        self.dni_database = dni_linear(dni_input_dim, dni_input_dim,  dni_hidden_size=dni_hidden_size)
        
    def get_scale(self) : 
        
        return self.scale_cls

        

        
    def get_pairwise_similarity(self, feat_query, feat_database) :
        '''
        Input :
            feat_query : 1, feat_dim
            feat_database : k, feat_dim
        
        Output :
            similarity_feat_query : similarity between all features and query features; (1 + k) * 1
            similarity_feat_database : similarity between all features and database features; (1 + k) * nb_img
            
        '''
        
        ## feat : (1 + k) * feat_dim
        feat = torch.cat([feat_query.view(-1, self.feat_dim), feat_database.view(-1, self.feat_dim)], dim=0)
        feat = F.normalize(feat, dim=feat.dim()-1, eps=1e-12)
        
        ## pairwise similarities
        similarity = torch.mm(feat, feat.transpose(0,1))
        similarity_query_query = similarity.narrow(0, 0, 1).narrow(1, 0, 1)
        similarity_database_database = similarity.narrow(0, 1, self.k).narrow(1, 1, self.k)
        similarity_query_database = similarity.narrow(0, 0, 1).narrow(1, 1, self.k)
        
        similarity_feat_query = torch.cat([similarity_query_query, similarity_query_database], dim=1).t()
        similarity_feat_database = torch.cat([similarity_query_database, similarity_database_database], dim=0)
        
        return similarity_feat_query, similarity_feat_database, similarity_query_database
        
    def subgraph_pairwise_similarity(self, similarity_feat_query, similarity_feat_database) : 
        '''
        Compute subgraph adjacency matrix, which is the input of the DNI
        
        '''
        
        with torch.no_grad() : 
            
            _, index_database = similarity_feat_database.topk(k=self.k, dim=1) ## sorting similarities with queries
            index_database = index_database + 1
            _, index_query = similarity_feat_query.topk(k=1, dim=1) ## sorting similarities with support
        
        
        index = torch.cat((index_query, index_database), dim = 1)
        similarity = torch.cat((similarity_feat_query, similarity_feat_database), dim = 1)
        
        dni_inputs = []
        
        for i in range(index.size()[1]) : 
            tmp_index = index[i] ## the i-th sample
            tmp_index = torch.cat((torch.tensor([i], dtype=torch.int64).cuda(), tmp_index), dim=0)
            
            tmp = torch.index_select(input=similarity, dim=0, index=tmp_index) 
            tmp = torch.index_select(input=tmp, dim=1, index=tmp_index)
            
            dni_inputs.append(tmp[self.triu_index_input].unsqueeze(0)) ## self.triu_index_input is defined at the beginning
            
        dni_inputs = torch.cat(dni_inputs, dim = 0)
        
        return dni_inputs
    
        
    def refine_feat(self, feat_query, feat_database):
        '''
        refine the query and database features
        '''
        
        
        for t in range(self.q_steps):
            
            ## update query
            similarity_query, similarity_database, _ = self.get_pairwise_similarity(feat_query, feat_database)
            inputs = self.subgraph_pairwise_similarity(similarity_query, similarity_database)
            
            grad_logit_query = self.dni_query(inputs) # B * n x nKnovel
            grad_query = torch.autograd.grad([inputs], [feat_query],
                                       grad_outputs=[grad_logit_query],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True, allow_unused=True)[0] # B x nKnovel x nFeat

            # perform GD
            
            feat_query  = feat_query  -  self.lr_dni * grad_query
            
            ## update query
            similarity_query, similarity_database, _ = self.get_pairwise_similarity(feat_query, feat_database)
            inputs = self.subgraph_pairwise_similarity(similarity_query, similarity_database)
            grad_logit_database = self.dni_database(inputs) # B * n x nKnovel
            grad_database = torch.autograd.grad([inputs], [feat_database],
                                       grad_outputs=[grad_logit_database],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True, allow_unused=True)[0] # B x nKnovel x nFeat

            # perform GD
            feat_database = feat_database - self.lr_dni * grad_database
            
        return feat_query, feat_database

                    
    def forward(self, feat_query=None, feat_database=None):
        
        feat_query, feat_database = self.refine_feat(feat_query, feat_database)
        _, _, similarity_query_database = self.get_pairwise_similarity(feat_query, feat_database)
        return similarity_query_database.squeeze()
        
    def forward_wo_refine(self, feat_query=None, feat_database=None):
        
        _, _, similarity_query_database = self.get_pairwise_similarity(feat_query, feat_database)
        return similarity_query_database.squeeze()
        
      
         
    
        

