import torch
import torch.nn.functional as F
import logging
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ['MSZ']


class MSZ:
    def __init__(self, args, data, model):

        self.args = args

        self.logger = logging.getLogger(args.logger_name)
        self.device, self.model = model.device, model.model

        # self.optimizer, self.scheduler = self._set_optimizer(args, data, self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, verbose=True, patience=args.wait_patience)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
        
        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)


    # def _set_optimizer(self, args, data, model):
        
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
        
    #     optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
    #     num_train_examples = len(data.train_data_index)
    #     num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
    #     num_warmup_steps= int(num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
    #     scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=num_warmup_steps,
    #                                                 num_training_steps=num_train_optimization_steps)
        
    #     return optimizer, scheduler
    

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    outputs = self.model(text_feats, video_feats, audio_feats, label_ids)

                    # fusion_feats = outputs['fusion_feats']
                    loss = outputs['loss']
                    
                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    
                    self.optimizer.step()
                    # self.scheduler.step()

            eval_outputs = self._get_outputs(args, mode = 'eval')
            self.scheduler.step(eval_outputs['loss'])
            eval_score = eval_outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
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
                
                outputs = self.model(text_feats, video_feats, audio_feats, label_ids)

                total_logits = torch.cat((total_logits, outputs['logits']))
                total_labels = torch.cat((total_labels, label_ids))
                 
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