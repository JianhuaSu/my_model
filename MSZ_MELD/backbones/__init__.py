from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MULT import MULT
from .FusionNets.MMIM import MMIM
from .FusionNets.MSZ import MSZ


text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'mmim': MMIM,
    'msz' : MSZ
}