class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)
        
    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """

        if args.text_backbone.startswith('bert'):

            hyper_parameters = {
                'need_aligned': True,
                'eval_monitor': 'acc',
                'train_batch_size': 32,
                'eval_batch_size': 16,
                'test_batch_size': 16,
                'wait_patience': 3,
                'num_train_epochs': 100,
                ################
                'beta_shift': [0.01],
                'dropout_prob': [0.2],
                'warmup_proportion': [0.05],
                'lr': [2e-5],
                'aligned_method': ['conv1d'],
                'weight_decay': [0.1],
                'scale':[2]
            } 
        else:
            raise ValueError('Not supported text backbone')        
        
     
        return hyper_parameters 