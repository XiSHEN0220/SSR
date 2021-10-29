# Code adapted from SIB: https://github.com/hushell/sib_meta_learn/blob/master/sib.py
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import dni_linear
import numpy as np

    
class SSR(nn.Module):
    """
    feat_dim: feature dimension at the input of classifier
    q_steps: number of update used in weights refinement
    nb_cls: nb of classes
    nb_qry: nb of queries
    lr_dni : learning rate for dni, 1e-3 is used in SIB
    dni_hidden_size: hidden dimension in dni
    """
    def __init__(self, feat_dim, q_steps, nb_cls, nb_qry, lr_dni = 1e-3, dni_hidden_size = 4096):
        super(SSR, self).__init__()

        self.feat_dim = feat_dim
        self.q_steps = q_steps
        self.nb_cls = nb_cls
        self.nb_qry = nb_qry
        self.lr_dni = lr_dni
        
        # scale of classifier
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        
        # dni grad_net
        dni_input_dim = (nb_cls + nb_qry + 1) * (nb_cls + nb_qry) // 2  - nb_qry * (nb_qry - 1) // 2
        tmp = torch.ones((1 + nb_cls + nb_qry), (1 + nb_cls + nb_qry)) ## full adjacency matrix
        tmp[1 + nb_cls : , 1 + nb_cls : ] = 0
        tmp = torch.triu(tmp, diagonal=1)
        tmp = 1 - tmp
        
        self.triu_index_input = (tmp == 0).cuda()
        
        
        ## we employ 2 dni for support and query set 
        self.dni_support = dni_linear(dni_input_dim, dni_input_dim, dni_hidden_size=dni_hidden_size)
        self.dni_query = dni_linear(dni_input_dim, dni_input_dim,  dni_hidden_size=dni_hidden_size)
        
        


    def get_supp_query(self, feat_support, feat_query) : 
        '''
        Input :
            feat_support : nb_cls, nb_shot, feat_dim
            feat_query : nb_qry, featdim
        
        Output :
            support : nb_cls, feat_dim
            query : nb_qry, feat_dim
        
        '''
        support = feat_support.mean(1) ## a simple average over the features
        query = feat_query
        
        return support, query
        
    def get_pairwise_similarity(self, feat_support, feat_query) :
        '''
        Input :
            feat_support : nb_cls, feat_dim
            feat_query : nb_qry, feat_dim
        
        Output :
            similarity_feat_support : similarity between all features and support features; (nb_cls + nb_qry) * nb_cls
            similarity_feat_query : similarity between all features and query features; (nb_cls + nb_qry) * nb_qry
            
        '''
        
        ## feat : (nb_cls + nb_qry) * feat_dim
        feat = torch.cat([feat_support.view(-1, self.feat_dim), feat_query.view(-1, self.feat_dim)], dim=0)
        feat = F.normalize(feat, dim=feat.dim()-1, eps=1e-12)
        
        ## pairwise similarities
        similarity = torch.mm(feat, feat.transpose(0,1))
        similarity_support_support = similarity.narrow(0, 0, self.nb_cls).narrow(1, 0, self.nb_cls)
        similarity_query_query = similarity.narrow(0, self.nb_cls, self.nb_qry).narrow(1, self.nb_cls, self.nb_qry)
        similarity_support_query = similarity.narrow(0, 0, self.nb_cls).narrow(1, self.nb_cls, self.nb_qry)
        
        similarity_feat_support = torch.cat([similarity_support_support, similarity_support_query], dim=1).t()
        similarity_feat_query = torch.cat([similarity_support_query, similarity_query_query], dim=0)
        
        return similarity_feat_support, similarity_feat_query, similarity_support_query
        
    def subgraph_pairwise_similarity(self, similarity_feat_support, similarity_feat_query) : 
        '''
        Compute subgraph adjacency matrix, which is the input of the DNI
        
        '''
        
        with torch.no_grad() : 
            
            _, index_query = similarity_feat_query.topk(k=self.nb_qry, dim=1) ## sorting similarities with queries
            index_query = index_query + similarity_feat_support.size()[1]
            _, index_support = similarity_feat_support.topk(k=self.nb_cls, dim=1) ## sorting similarities with support
        
        
        index = torch.cat((index_support, index_query), dim = 1)
        similarity = torch.cat((similarity_feat_support, similarity_feat_query), dim = 1)
        
        dni_inputs = []
        
        for i in range(index.size()[1]) : 
            tmp_index = index[i] ## the i-th sample
            tmp_index = torch.cat((torch.tensor([i], dtype=torch.int64).cuda(), tmp_index), dim=0)
            
            tmp = torch.index_select(input=similarity, dim=0, index=tmp_index) 
            tmp = torch.index_select(input=tmp, dim=1, index=tmp_index)
            
            dni_inputs.append(tmp[self.triu_index_input].unsqueeze(0)) ## self.triu_index_input is defined at the beginning
            
        dni_inputs = torch.cat(dni_inputs, dim = 0)
        
        return dni_inputs
    
        
    def refine_feat(self, feat_supp, feat_query):
        '''
        refine the support and query feature
        '''
        
        
        supp, query = self.get_supp_query(feat_supp, feat_query)
        
        for t in range(self.q_steps):
            
            ## update support
            similarity_supp, similarity_query, _ = self.get_pairwise_similarity(supp, query)
            inputs = self.subgraph_pairwise_similarity(similarity_supp, similarity_query)
            
            grad_logit_support = self.dni_support(inputs) # B * n x nKnovel
            grad_support = torch.autograd.grad([inputs], [supp],
                                       grad_outputs=[grad_logit_support],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True, allow_unused=True)[0] # B x nKnovel x nFeat

            # perform GD
            
            supp  = supp  -  self.lr_dni * grad_support
            
            ## update query
            similarity_supp, similarity_query, _ = self.get_pairwise_similarity(supp, query)
            inputs = self.subgraph_pairwise_similarity(similarity_supp, similarity_query)
            grad_logit_query = self.dni_query(inputs) # B * n x nKnovel
            grad_query = torch.autograd.grad([inputs], [query],
                                       grad_outputs=[grad_logit_query],
                                       create_graph=True, retain_graph=True,
                                       only_inputs=True, allow_unused=True)[0] # B x nKnovel x nFeat

            # perform GD
            query = query - self.lr_dni * grad_query
            
        return supp, query

                    
    def forward(self, feat_suppoprt=None, feat_query=None):
        
        feat_suppoprt, feat_query = self.refine_feat(feat_suppoprt, feat_query)
        _, _, similarity_supp_query = self.get_pairwise_similarity(feat_suppoprt, feat_query)
        cls_scores = similarity_supp_query * self.scale_cls
        return cls_scores.t()
        
    def forward_wo_refine(self, feat_suppoprt=None, labels_supp=None, feat_query=None):
        
        feat_suppoprt, feat_query = self.get_supp_query(feat_suppoprt, feat_query)
        _, _, similarity_supp_query = self.get_pairwise_similarity(feat_suppoprt, feat_query)
        cls_scores = similarity_supp_query * self.scale_cls
        return cls_scores.t()
         
    
        

