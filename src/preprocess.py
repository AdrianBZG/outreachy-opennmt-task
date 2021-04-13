from config import defaults
from src.pipeline import tokenize, readers

def setup_dataset(name:str):
    if name not in defaults.datasets:
        raise NotADirectoryError

    datareader = None
    if name == 'toy-ende':
        datareader = readers.ToyENDEReader()
    elif name == 'rapid':
        datareader = readers.RapidSETRead()
    else:
        raise NotImplementedError
        
    dataset = datareader._parse_data()
    return dataset