import numpy as np
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
from scipy.special import logsumexp

# def func(args, inputs):
#     w = inputs['w']
#     w = w.data.cpu().numpy()
#     w_norm = w / np.linalg.norm(w, axis=1, keepdims=True)

#     train_feats = inputs['train_feats']
#     train_feats_norm = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)

#     # 计算余弦相似度，代替原有的线性变换逻辑
#     train_logits = np.dot(train_feats_norm, w_norm.T) * args.scale

#     u = -np.matmul(pinv(w_norm), np.zeros_like(w_norm).mean(axis=1))

#     ec = EmpiricalCovariance(assume_centered=True)
#     ec.fit(train_feats_norm - u)
#     eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
#     NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[args.num_labels:]]).T)

#     vlogit_ind_train = norm(np.matmul(train_feats_norm - u, NS), axis=-1)
#     alpha = train_logits.max(axis=-1).mean() / vlogit_ind_train.mean()

#     features = inputs['y_feat']
#     features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
#     logit = np.dot(features_norm, w_norm.T) * args.scale
#     vlogit = norm(np.matmul(features_norm - u, NS), axis=-1) * alpha
#     energy = logsumexp(logit, axis=-1)
#     scores = -vlogit + energy

#     return scores

def func(args, inputs):
    
    w = inputs['w']
    w = w.data.cpu().numpy()
    b = inputs['b']
    b = b.data.cpu().numpy()
    
    train_feats = inputs['train_feats']
    train_logits = train_feats @ w.T + b

    u = -np.matmul(pinv(w), b)
            
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feats - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[args.num_labels:]]).T)
    # 主子空间p的正交补

    vlogit_ind_train = norm(np.matmul(train_feats - u, NS), axis=-1)
    alpha = train_logits.max(axis=-1).mean() / vlogit_ind_train.mean()

    features = inputs['y_feat']
    logit = features @ w.T + b
    vlogit = norm(np.matmul(features - u, NS), axis=-1) * alpha
    energy = logsumexp(logit, axis=-1)
    scores = -vlogit + energy

    return scores