import torch
import torch.nn.functional as F
import logging
import numpy as np
import os
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import get_dataloader
from torch.utils.data import DataLoader
from utils.metrics import AverageMeter, Metrics, OOD_Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from ood_detection import ood_detection_map
from backbones.FusionNets.MSZ import MMEncoder, MLP_head
from sklearn.metrics import accuracy_score

__all__ = ['MSZ']

class MSZ:

    def __init__(self, args, data):
             
        self.logger = logging.getLogger(args.logger_name)
        
        self.device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        self.args = args
        
        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        self.seq_train_dataloader = DataLoader(mm_data['train'], shuffle=False, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
        
        self.ood_detection_func = ood_detection_map[args.ood_detection_method]
        self.ood_metrics = OOD_Metrics(args)
        self.metrics = Metrics(args)

        self.mm_encoder = MMEncoder(args).to(self.device)
        self.multiclass_classifier = MLP_head(args, args.num_labels).to(self.device)
        
        if self.args.ablation != 'no_binary':
            
            self.binary_classifier = MLP_head(args, 2).to(self.device)
            param_optimizer = list(self.mm_encoder.named_parameters()) + \
                            list(self.multiclass_classifier.named_parameters()) + \
                            list(self.binary_classifier.named_parameters())
                            
        else:
            param_optimizer = list(self.mm_encoder.named_parameters()) + \
                            list(self.multiclass_classifier.named_parameters())

        self.optimizer, self.scheduler = self._set_optimizer(args, args.lr, param_optimizer, args.weight_decay, args.warmup_proportion)
        
        self.criterion = nn.CrossEntropyLoss()
        
        if not args.train:
            self.best_eval_score = 0
            self.mm_encoder = restore_model(self.mm_encoder, os.path.join(args.model_output_path, 'mm_encoder'), self.device)
            self.multiclass_classifier = restore_model(self.multiclass_classifier, os.path.join(args.model_output_path, 'multiclass_classifier'), self.device)
            if self.args.ablation != 'no_binary':
                self.binary_classifier = restore_model(self.binary_classifier, os.path.join(args.model_output_path, 'binary_classifier'), self.device)


    def _set_optimizer(self, args, lr, param_optimizer, weight_decay, warmup_proportion):
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = lr, correct_bias = True)
        
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler
    
    
    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            
            if self.args.ablation != 'no_binary':
                self.mm_encoder.train()
                self.binary_classifier.train()
                binary_loss_record = AverageMeter()

                for step, batch in enumerate(tqdm(self.train_dataloader, desc = 'Binary')):

                    text_feats = batch['text_feats'].to(self.device)
                    video_feats = batch['video_feats'].to(self.device)
                    audio_feats = batch['audio_feats'].to(self.device)
                    label_ids = batch['label_ids'].to(self.device)

                    with torch.set_grad_enabled(True):

                        mix_fusion_feats, mixed_labels = self.mm_encoder(text_feats, video_feats, audio_feats, label_ids, ood_sampling = True)
                        binary_logits = self.binary_classifier(mix_fusion_feats)
                        binary_labels = mixed_labels['binary']

                        loss = self.criterion(binary_logits, binary_labels)

                        self.optimizer.zero_grad()
                    
                        loss.backward()
                        binary_loss_record.update(loss.item(), binary_labels.size(0))


                        self.optimizer.step()
                        self.scheduler.step()

                outputs = self._get_binary_outputs(args, self.eval_dataloader)
                binary_eval_score = outputs['binary_acc']
                

                self.binary_classifier.eval()
                
            self.mm_encoder.train()
            self.multiclass_classifier.train()
            multiclass_loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Multi-class")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):
                    
                    mix_fusion_feats, _ = self.mm_encoder(text_feats, video_feats, audio_feats, label_ids, ood_sampling = True)

                    id_num = len(label_ids)

                    if self.args.ablation != 'no_binary':
                        binary_logits = self.binary_classifier(mix_fusion_feats)
                        probs = F.softmax(binary_logits, dim = 1)
                        scores = []

                        for i in range(len(binary_logits)):
                            if i < id_num:
                                scores.append(probs[i][1])
                            else:
                                scores.append(probs[i][0])
                        
                        binary_scores = torch.tensor(scores).to(self.device)
                        logits = self.multiclass_classifier(mix_fusion_feats, binary_scores = binary_scores)
                    else: 
                        logits = self.multiclass_classifier(mix_fusion_feats) 

                    id_logits = logits[:id_num]
                    loss = self.criterion(id_logits, label_ids)

                    self.optimizer.zero_grad()
                    loss.backward()
                    multiclass_loss_record.update(loss.item(), label_ids.size(0))
                    
                    self.optimizer.step()
                    self.scheduler.step()

            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs['acc']

            if self.args.ablation != 'no_binary':
                eval_results = {
                    'binary_train_loss': round(binary_loss_record.avg, 4),
                    'binary_eval_score': round(binary_eval_score, 4),

                    'train_loss': round(multiclass_loss_record.avg, 4),
                    'eval_score': round(eval_score, 4),

                    'best_eval_score': round(early_stopping.best_score, 4),
                    
                }
            else:
                eval_results = {

                    'train_loss': round(multiclass_loss_record.avg, 4),
                    'eval_score': round(eval_score, 4),

                    'best_eval_score': round(early_stopping.best_score, 4),
                    
                }
                                
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            if self.args.ablation != 'no_binary':
                early_stopping(eval_score, self.mm_encoder, self.multiclass_classifier, self.binary_classifier)
            else:
                early_stopping(eval_score, self.mm_encoder, self.multiclass_classifier)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.mm_encoder = early_stopping.best_model
        self.multiclass_classifier = early_stopping.best_multiclass_head

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            
            os.makedirs(os.path.join(args.model_output_path, 'mm_encoder'), exist_ok = True)
            save_model(self.mm_encoder, os.path.join(args.model_output_path, 'mm_encoder'))  

            os.makedirs(os.path.join(args.model_output_path, 'multiclass_classifier'), exist_ok = True)
            save_model(self.multiclass_classifier, os.path.join(args.model_output_path, 'multiclass_classifier'))  

            if self.args.ablation != 'no_binary':
                os.makedirs(os.path.join(args.model_output_path, 'binary_classifier'), exist_ok = True)
                save_model(self.binary_classifier, os.path.join(args.model_output_path, 'binary_classifier'))  

    def _get_binary_outputs(self, args, dataloader, show_results = False):

        self.binary_classifier.eval()
        self.mm_encoder.eval()

        total_binary_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_binary_logits = torch.empty((0, 2)).to(self.device)
        total_features = torch.empty((0, args.hidden_dims)).to(self.device)


        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):

                mix_fusion_feats, mixed_labels = self.mm_encoder(text_feats, video_feats, audio_feats, label_ids, ood_sampling = True)
                binary_logits = self.binary_classifier(mix_fusion_feats)
                binary_labels = mixed_labels['binary']

                total_binary_logits = torch.cat((total_binary_logits, binary_logits))
                total_binary_labels = torch.cat((total_binary_labels, binary_labels))
                total_features = torch.cat((total_features, mix_fusion_feats))

        binary_scores = F.softmax(total_binary_logits.detach(), dim = 1)
        _, binary_preds = binary_scores.max(dim = 1)
        y_binary_pred = binary_preds.cpu().numpy()
        y_binary_labels = total_binary_labels.cpu().numpy()
        binary_acc = accuracy_score(y_binary_pred, y_binary_labels)

        outputs = {}
            
        outputs.update(
            {
                'y_true': y_binary_labels,
                'y_pred': y_binary_pred,
                'binary_acc': binary_acc,
            }
        )

        return outputs

    def _get_outputs(self, args, dataloader, show_results = False, test_ind = False):
        
        if self.args.ablation != 'no_binary':
            self.binary_classifier.eval()
        self.mm_encoder.eval()
        self.multiclass_classifier.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.hidden_dims)).to(self.device)

        total_binary_scores = torch.empty(0,dtype=torch.float64).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)

            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):

                mix_fusion_feats, _ = self.mm_encoder(text_feats, video_feats, audio_feats, label_ids, ood_sampling = False)
                
                if self.args.ablation != 'no_binary':

                    binary_logits = self.binary_classifier(mix_fusion_feats)
                    binary_scores = F.softmax(binary_logits, dim = 1)[:, 1]
                    logits = self.multiclass_classifier(mix_fusion_feats, binary_scores = binary_scores) 
                else:
                    logits = self.multiclass_classifier(mix_fusion_feats) 

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))

                total_features = torch.cat((total_features, mix_fusion_feats))
                
                if self.args.ablation != 'no_binary':
                    total_binary_scores = torch.cat((total_binary_scores, binary_scores))
                    y_score = total_binary_scores.cpu().numpy()

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()

        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        if test_ind:
            outputs = self.metrics(y_true[y_true != args.ood_label_id], y_pred[y_true != args.ood_label_id])
        else:
            outputs = self.metrics(y_true, y_pred, show_results = show_results)
            
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat,
            }
        )
        
        return outputs

    def _test(self, args):
        
        test_results = {}
        
        ind_test_results = self._get_outputs(args, self.test_dataloader, show_results = True, test_ind = True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)
        
        if args.ood:
            
            tmp_outputs = self._get_outputs(args, self.test_dataloader)

            
            if args.ood_detection_method in ['residual', 'ma', 'vim']:
                ind_train_outputs = self._get_outputs(args, self.train_dataloader)
                
                tmp_outputs['train_feats'] = ind_train_outputs['y_feat']
                tmp_outputs['train_labels'] = ind_train_outputs['y_true']
                
                w, b = self.multiclass_classifier.vim()

                tmp_outputs['w'] = w
                tmp_outputs['b'] = b


            scores = self.ood_detection_func(args, tmp_outputs)
            binary_labels = np.array([1 if x != args.ood_label_id else 0 for x in tmp_outputs['y_true']])

            ood_test_scores = self.ood_metrics(scores, binary_labels, show_results = True) 
            test_results.update(ood_test_scores)
        
        return test_results
