import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, centroids_cal


class PretrainManager:
    
    def __init__(self, args, data, model, logger_name = 'Detection'):

        self.logger = logging.getLogger(logger_name)

        self.set_model_optimizer(args, data, model)
        
        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = loss_map[args.loss_fct]  
        self.contrast_criterion = loss_map['SupConLoss'] 
        self.triplet_criterion = nn.TripletMarginLoss(margin=args.margin, p=args.p)

        self.centroids = None
        self.best_eval_score = None

        if args.pretrain or (not os.path.exists(args.model_output_dir)):
            self.logger.info('Pre-training Begin...')

            if args.backbone == 'bert_disaware':
                self.train_disaware(args, data)
            else:
                self.train_plain(args, data)

            self.logger.info('Pre-training finished...')
                
        else:
            self.model = restore_model(self.model, args.model_output_dir)

    def set_model_optimizer(self, args, data, model):
    
        self.model = model.set_model(args, 'bert')  
        self.optimizer, self.scheduler = model.set_optimizer(self.model, data.dataloader.num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.device = model.device

    def train_plain(self, args, data):

        wait = 0
        best_model = None
        best_eval_score = 0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    n_loss, pooled_output, n_logits, c_logits = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train", loss_fct=self.loss_fct)
                    
                    batch_size = pooled_output.shape[0]
                    
                    # 原有的SupConLoss计算
                    norm_logits = F.normalize(n_logits)
                    norm_logits_aug = F.normalize(c_logits)
                    t_label_ids = label_ids.expand(batch_size, batch_size)
                    mask = torch.eq(t_label_ids, t_label_ids.T).long()
                    logits_mask = torch.scatter(
                        mask,
                        0,
                        torch.arange(batch_size).unsqueeze(0).to(self.device),
                        1
                    )
                    contrastive_logits = torch.cat((norm_logits.unsqueeze(1), norm_logits_aug.unsqueeze(1)), dim=1)
                    supcon_loss = self.contrast_criterion(contrastive_logits, mask=logits_mask, temperature=0.3, device=self.device)
                    
                    # 新增的三元组损失计算
                    dist_matrix = torch.cdist(pooled_output, pooled_output)
                    anchors, positives, negatives = [], [], []
                    for i in range(batch_size):
                        anchor = pooled_output[i]
                        anchor_label = label_ids[i]
                        
                        # 选择同类别中最远的样本作为正样本
                        pos_candidates = (label_ids == anchor_label) & (torch.arange(batch_size).to(self.device) != i)
                        if pos_candidates.sum() > 0:
                            pos_index = dist_matrix[i][pos_candidates].argmax()
                            positive = pooled_output[pos_candidates][pos_index]
                        else:
                            positive = anchor  # 如果没有其他同类样本，使用自身
                        
                        # 选择不同类别中最近的样本作为负样本
                        neg_candidates = label_ids != anchor_label
                        if neg_candidates.sum() > 0:
                            neg_index = dist_matrix[i][neg_candidates].argmin()
                            negative = pooled_output[neg_candidates][neg_index]
                        else:
                            negative = pooled_output[(torch.arange(batch_size) != i).nonzero().squeeze()][0]  # 随机选择一个不同的样本
                        
                        anchors.append(anchor)
                        positives.append(positive)
                        negatives.append(negative)
                    
                    anchors = torch.stack(anchors)
                    positives = torch.stack(positives)
                    negatives = torch.stack(negatives)
                    
                    triplet_loss = self.triplet_criterion(anchors, positives, negatives)
                    
                    # 结合所有损失
                    loss = args.loss_weight * n_loss + \
                           args.supcon_loss_weight * supcon_loss + \
                           args.triplet_loss_weight * triplet_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)

    def train_disaware(self, args, data):

        wait = 0
        best_model = None
        best_centroids = None
        best_eval_score = 0
        args.device = self.device
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.centroids = centroids_cal(self.model, args, data, self.train_dataloader, self.device)
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.set_grad_enabled(True):
                    
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train", loss_fct = self.loss_fct, centroids = self.centroids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps

            y_true, y_pred = self.get_outputs(args, data, mode = 'eval')
            eval_score = round(f1_score(y_true, y_pred, average = 'macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                best_centroids = copy.copy(self.centroids)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        self.centroids = best_centroids
        self.best_eval_score = best_eval_score

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)       
        

    def get_outputs(self, args, data, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):

                if args.backbone == 'bert_disaware':
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask, centroids = self.centroids, labels = label_ids, mode = mode)
                else:    
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask, mode = mode)

                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats

        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred
