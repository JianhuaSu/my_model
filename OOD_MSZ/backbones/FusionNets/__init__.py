from .MAG_BERT import MAG_BERT
from .MISA import MISA
from .MMIM import MMIM
from .MULT import MULT
from .MSZ import MMEncoder

multimodal_methods_map = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mmim': MMIM, 
    'mult': MULT,
    'msz': MMEncoder,
}