import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics
from loss.angular import AngularPenaltySMLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import optim

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ['MSZ']


class MSZ:
    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.cosface = AngularPenaltySMLoss(args, loss_type='cosface')

        self.adamw_outputs = self._set_optimizer(args, data, self.model)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _set_optimizer(self, args, data, model):

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.model.text_model.named_parameters())
        audio_params = list(model.model.audio_model.named_parameters())
        video_params = list(model.model.video_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        specific_params = [p for n,p in list(model.model.named_parameters()) if  'Specific' in n]
        specific_params += [j for i,j in self.cosface.Specific_fc.named_parameters()]
        corrector_params = [p for n,p in list(model.model.named_parameters()) if  'corrector' in n]
        model_params_other = [p for n, p in list(model.model.named_parameters()) if 'text_model' not in n and 'audio_model' not in n and 'video_model' not in n and 'Specific' not in n]
        
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other},
            {'params': specific_params,'weight_decay':self.args.weight_decay_specific,'lr': self.args.learning_rate_specific},
            {'params': corrector_params,'weight_decay':self.args.weight_decay_corrector,'lr': self.args.learning_rate_corrector}
        ]

        optimizer_all = AdamW(optimizer_grouped_parameters[:5], correct_bias=False)
        optimizer_specific = AdamW([optimizer_grouped_parameters[5]], correct_bias=False)
        optimizer_corrector = AdamW([optimizer_grouped_parameters[-1]], correct_bias=False)

        num_train_examples = len(data.train_data_index)
        num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler_all = get_linear_schedule_with_warmup(optimizer_all,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        scheduler_specific = get_linear_schedule_with_warmup(optimizer_specific,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        scheduler_corrector = get_linear_schedule_with_warmup(optimizer_corrector,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        outputs = {
            'all' : [optimizer_all, scheduler_all],
            'specific' : [optimizer_specific, scheduler_specific],
            'corrector' : [optimizer_corrector, scheduler_corrector],
        }

        return outputs
    

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            # main
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration-main")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    outputs = self.model(text_feats, video_feats, audio_feats, label_ids)

                    # all
                    loss = self.criterion(outputs['M'], label_ids)

                    self.adamw_outputs['all'][0].zero_grad()
                    self.adamw_outputs['specific'][0].zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    
                    self.adamw_outputs['all'][0].step()
                    self.adamw_outputs['all'][1].step()

                    feature_t = self.model.model.Specific_t(outputs['t'].detach())
                    feature_v = self.model.model.Specific_v(outputs['v'].detach())
                    feature_a = self.model.model.Specific_a(outputs['a'].detach())
                    feature_m = outputs['feature_m'].detach()

                    # specific
                    s_feature = torch.cat([feature_m,feature_t,feature_v,feature_a],dim=0)
                    s_label =  torch.cat([torch.ones_like(torch.mean(feature_m,dim=1))*i for i in range(4)],dim=0).long()
                    specific_loss = self.cosface(s_feature, s_label)
                    specific_loss.backward()
                    self.adamw_outputs['specific'][0].step()
                    self.adamw_outputs['specific'][1].step()

            # bise
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration-bise")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    self.adamw_outputs['corrector'][0].zero_grad()
                    outputs = self.model(text_feats, video_feats, audio_feats)
                    true_label = outputs['M'].detach()
                    feature_t = self.model.model.Specific_t(outputs['t'].detach())
                    feature_v = self.model.model.Specific_v(outputs['v'].detach())
                    feature_a = self.model.model.Specific_a(outputs['a'].detach())
                    fusion_feature = torch.cat([feature_t,feature_a,feature_v],dim=1)
                    new_feature = self.model.model.corrector_layer_1(fusion_feature)
                    new_feature = self.model.model.corrector_layer_2(new_feature)
                    corrector_label = self.model.model.corrector_layer_3(new_feature)

                    if args.use_tanh:
                        corrector_label = torch.tanh(corrector_label)

                    # 纠正标签
                    final_label = true_label + args.scale * corrector_label
                    loss = self.criterion(final_label, label_ids)
                    loss.backward()

                    self.adamw_outputs['corrector'][0].step()
                    self.adamw_outputs['corrector'][1].step()

            eval_outputs = self._get_outputs(args, mode = 'eval')
            eval_score = eval_outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
         
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model   
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   


    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
                
                outputs = self.model(text_feats, video_feats, audio_feats)

                true_label = outputs['M'].detach()
                feature_t = self.model.model.Specific_t(outputs['t'].detach())
                feature_v = self.model.model.Specific_v(outputs['v'].detach())
                feature_a = self.model.model.Specific_a(outputs['a'].detach())
                fusion_feature = torch.cat([feature_t,feature_a,feature_v],dim=1)
                new_feature = self.model.model.corrector_layer_1(fusion_feature)
                new_feature = self.model.model.corrector_layer_2(new_feature)
                corrector_label = self.model.model.corrector_layer_3(new_feature)
                final_label = true_label + args.scale * corrector_label

                total_logits = torch.cat((total_logits, final_label))
                total_labels = torch.cat((total_labels, label_ids))
 
                loss = self.criterion(outputs['M'], label_ids)
                loss_record.update(loss.item(), label_ids.size(0))
                
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
        outputs.update({'loss': loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs


    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results=True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)
    
        return test_results