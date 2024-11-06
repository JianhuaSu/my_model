from .FeatureNets import BERTEncoder, RoBERTaEncoder

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'roberta-base': RoBERTaEncoder
                }