    
class Param():
    
    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):

        if args.text_backbone.startswith('bert'):
            
            hyper_parameters = {
                
                'need_aligned': True,
                'freeze_parameters': False,
                'eval_monitor': 'acc',
                
                # mananger
                'train_batch_size': 32, # [32, 64, 128]
                'eval_batch_size': 16,
                'num_train_epochs': 100,
                
                # 优化器
                'lr': [2e-5], #3e-5
                'weight_decay': [0.1],
                'wait_patience': 5,
                'warmup_proportion': [0.1], 

                # msz
                'hidden_dims':[768],
                "num_attention_heads": [4],
                "attention_probs_dropout_prob":[0.3],
                "hidden_dropout_prob": [0.1],
                "intermediate_size": [768],  
                "num_hidden_layers": [4],
                'max_position_embeddings': [900],
                "position_embedding_type": "absolute",
                'batch_first': True,
                'aligned_method': ['ctc'],

                # sampler
                'sampler':['dir'],
                'multiple_ood': [1],
                'select_number_min': [2],
                'select_number_max': [3],
                'alpha': [4],

                # MLP
                'mlp_hidden_size': [256],
                'mlp_dropout': [0.3],
                'scale': [16],
                'ablation':['no_binary']

            }
       
        return hyper_parameters
    
    
# class Param():
    
#     def __init__(self, args):

#         self.hyper_param = self._get_hyper_parameters(args)

#     def _get_hyper_parameters(self, args):

#         if args.text_backbone.startswith('bert'):
            
#             hyper_parameters = {
                
#                 'need_aligned': True,
#                 'freeze_parameters': False,
#                 'eval_monitor': 'acc',
                
#                 # mananger
#                 'train_batch_size': 32, # [32, 64, 128]
#                 'eval_batch_size': 16,
#                 'num_train_epochs': 100,
                
#                 # 优化器
#                 'lr': [2e-5], #3e-5
#                 'weight_decay': [0.1],
#                 'wait_patience': 5,
#                 'warmup_proportion': [0.01], 

#                 # msz
#                 'hidden_dims':[768],
#                 "num_attention_heads": [8],
#                 "attention_probs_dropout_prob":[0.3],
#                 "hidden_dropout_prob": [0.2],
#                 "intermediate_size": [768],  
#                 "num_hidden_layers": [4],
#                 'max_position_embeddings': [900],
#                 "position_embedding_type": "absolute",
#                 'batch_first': True,
#                 'aligned_method': ['ctc'],

#                 # sampler
#                 'sampler':'beta',
#                 'multiple_ood': [1],
#                 'select_number_min': [2],
#                 'select_number_max': [3],
#                 'alpha': [4],

#                 # MLP
#                 'mlp_hidden_size': [256],
#                 'mlp_dropout': [0.3],
#                 'scale': [8],
#                 'ablation':['no_binary']

#             }
       
#         return hyper_parameters
    