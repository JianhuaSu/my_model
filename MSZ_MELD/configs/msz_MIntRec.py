class Param():
    
    def __init__(self, args):
        
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
        }
        return common_parameters

    def _get_hyper_parameters(self, args):

        hyper_parameters = {
            # attention
            'hidden_dims':[768],
            "num_attention_heads": [12],
            "attention_probs_dropout_prob":[0.2],
            "hidden_dropout_prob": [0.2],
            "intermediate_size": [768],  
            "num_hidden_layers": [8],
            'max_position_embeddings': [768],
            "position_embedding_type": "absolute",
            'batch_first': True,

            # manager
            'dropout':[0.1],
            'aligned_method': ['conv1d'],
            'num_train_epochs': 100,
            'lr': [8e-6],
            'lr_mode' : 'max',
            'patience' : [8], # 8 
            'factor' : [0.1], # 0.1
            
            # 'ablation':['no_fatt']
        }
        return hyper_parameters