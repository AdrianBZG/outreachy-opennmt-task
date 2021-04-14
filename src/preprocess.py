from config import defaults, genconfig

from src.utils.dataset import Dataset
from src.pipeline import readers, vocab

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
    genconfig.gen_yaml_config(dataset)

    return dataset


def setup_vocab(ds: Dataset):
    options, unknown = vocab._build_vocabulary(ds)

    return