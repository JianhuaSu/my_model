from .MAG_BERT.manager import MAG_BERT
from .MISA.manager import MISA
from .TEXT.manager import TEXT
from .MMIM.manager import MMIM
from .MULT.manager import MULT
from .MSZ.manager import MSZ


method_map = {
    
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'text': TEXT,
    'mmim': MMIM,
    'mult': MULT,
    'msz': MSZ

}