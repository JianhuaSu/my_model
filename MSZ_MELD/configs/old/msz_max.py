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
            # transform
            'nheads': 12,
            'encoder_layers': 6,
            'attn_dropout': 0.1,  #0.2
            'relu_dropout': 0.1,
            'embed_dropout': 0.1,
            'res_dropout': 0.1,
            'attn_mask': False,

            # attention
            "hidden_size": 768,  # default is 768
            "num_attention_heads": 12,  # default is 12
            "attention_probs_dropout_prob":0.1,
            "hidden_dropout_prob": 0.1,
            "intermediate_size": 768,  
            "num_hidden_layers": 6,
            'max_position_embeddings': 768,
            "position_embedding_type": "absolute",
            'batch_first': True,

            # model
            'hidden_dims':768,
            'att_dropouts':(0.2, 0.2, 0.2),
            'spe_dropouts':(0.2, 0.1, 0.1),
            'aligned_method': 'conv1d',
            'grad_clip':2,

            # manager
            'num_train_epochs': 100,
            'warmup_proportion': 0.1,
            'lr': 3e-6,
            'weight_decay': 0.01

        }
        return hyper_parameters