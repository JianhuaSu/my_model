import torch
from torch import nn
import torch.nn.functional as F


# class BoundaryLoss(nn.Module):
#     """
#     Deep Open Intent Classification with Adaptive Decision Boundary.
#     https://arxiv.org/pdf/2012.10209.pdf
#     """
#     def __init__(self, num_labels=10, feat_dim=2, device = None):
#         super(BoundaryLoss, self).__init__()
#         self.num_labels = num_labels
#         self.feat_dim = feat_dim
#         self.delta = nn.Parameter(torch.randn(num_labels).to(device))
#         nn.init.normal_(self.delta)
        
#     def forward(self, pooled_output, centroids, labels):
        
#         delta = F.softplus(self.delta)
#         c = centroids[labels]
#         d = delta[labels]
#         x = pooled_output
        
#         euc_dis = torch.norm(x - c,2, 1).view(-1)
#         pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
#         neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
        
#         pos_loss = (euc_dis - d) * pos_mask
#         neg_loss = (d - euc_dis) * neg_mask
        
#         loss = pos_loss.mean() + neg_loss.mean()

#         return loss, delta 
    
    

def euclidean_metric(a, b, cosine=False):
    if cosine:
        return a @ b.T
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -torch.sqrt(((a - b)**2).sum(dim=2))
    return logits

def cosine_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = torch.cosine_similarity(a, b, dim=2)
    return logits


def cosine_distance(a, b, dim=1):
    cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
    output = cos(a, b)
    return output


class BoundaryLoss(nn.Module):

    def __init__(self, args, num_labels=10, feat_dim=2, neg=False, device = None):
        
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        # self.delta = nn.Parameter(torch.randn(num_labels).cuda() * 0.2)
        self.delta = nn.Parameter(torch.randn(num_labels).to(device))
        self.neg = neg
        self.safe1= args.safe1
        self.safe2= args.safe2
        self.neg_weight = args.neg_weight
        nn.init.normal_(self.delta)
        
    def forward(self, pooled_output, centroids, labels):
        
        delta = F.softplus(self.delta)

        pooled_output_shape = pooled_output.shape
        pooled_output = pooled_output.reshape(-1, 2, pooled_output_shape[-1])
        pooled_output, neg_output = pooled_output[:, 0, :], pooled_output[:, 1, :]

        if self.neg:
            c = centroids[labels]
            d = delta[labels]
            safe2 = self.safe2
            safe1 = self.safe1
            x = pooled_output
            
            euc_dis = torch.norm(x - c,2, 1).view(-1)
            n_euc_dis = torch.norm(neg_output - c,2, 1).view(-1)

            pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
            neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

            pos_loss = (euc_dis - d) * pos_mask
            neg_loss = (d - euc_dis) * neg_mask

            # n_mask = (n_euc_dis > d).type(torch.cuda.FloatTensor)
            n_pos_mask = (n_euc_dis > d + safe2).type(torch.cuda.FloatTensor)
            n_neg_mask = (n_euc_dis - safe1 < d).type(torch.cuda.FloatTensor)

            n_pos_loss = (n_euc_dis - d - safe2) * n_pos_mask
            n_neg_loss = (d - n_euc_dis + safe1) * n_neg_mask

            loss = pos_loss.mean() + neg_loss.mean() + (n_pos_loss.mean()+ n_neg_loss.mean() ) * self.neg_weight
            
        else:
            c = centroids[labels]
            d = delta[labels]
            x = pooled_output

            euc_dis = torch.norm(x - c,2, 1).view(-1)

            pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
            neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

            pos_loss = (euc_dis - d) * pos_mask
            neg_loss = (d - euc_dis) * neg_mask

            loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta 