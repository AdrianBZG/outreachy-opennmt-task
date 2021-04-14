from config import defaults
from src.pipeline import tokenize, readers

def setup_dataset(name:str):
    if name not in defaults.datasets:
        raise NotADirectoryError

    datareader = None
    if name == 'toy-ende':
        datareader = readers.ToyENDEReader(defaults.datapaths[name])
    elif name == 'rapid2016':
        datareader = readers.RapidSETReader(defaults.datapaths[name])
    elif name == 'wiki':
        raise NotImplementedError
        # datareader = readers.RapidSETReader(defaults.datapaths[name])
    else:
        raise NotImplementedError
        
    dataset = datareader._parse_data()
    return dataset