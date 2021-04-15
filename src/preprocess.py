from config import defaults, genconfig

from src.utils.dataset import Dataset
from src.pipeline import readers, vocab

def setup_dataset(name: str):
    if name not in defaults.datasets:
        raise NotADirectoryError

    datareader = None
    if name == 'toy-ende':
        datareader = readers.ToyENDEReader(defaults.datapaths[name])
    elif name == 'rapid2016':
        datareader = readers.RapidSETReader(defaults.datapaths[name])
    elif name == 'wiki':
        datareader = readers.WikiTitlesReader(defaults.datapaths[name])
    else:
        raise NotImplementedError
        
    dataset = datareader._parse_data()
    genconfig.gen_yaml_config(dataset)

    return dataset


def setup_vocab(ds: Dataset):
    options, unknown = vocab._build_vocabulary(ds)
    vocab_fields = vocab._build_fields()

    # src_text_field = vocab_fields["src"].base_field
    # src_vocab = src_text_field.vocab
    # src_padding = src_vocab.stoi[src_text_field.pad_token]

    # tgt_text_field = vocab_fields['tgt'].base_field
    # tgt_vocab = tgt_text_field.vocab
    # tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    return vocab_fields