from config import defaults
from pipeline import build_vocab, tokenize

def setup_dataset(name:str):
    if name not in defaults.datasets:
        raise NotADirectoryError
        
    pass