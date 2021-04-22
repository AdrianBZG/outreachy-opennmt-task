from os import path
from collections import defaultdict, Counter
from config import defaults
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.bin.build_vocab import build_vocab_main
from onmt.inputters import inputter

from src.utils.dataset import Dataset


def build_vocabulary(ds: Dataset) -> None:
    base_args = ([
        "-config", f"{path.join(ds.path, 'config.yaml')}", 
        "-n_sample", "10000"
    ])

    parser = ArgumentParser(description='vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only=True)

    options, unknown = parser.parse_known_args(base_args)
    build_vocab_main(options)

    return options, unknown


def build_fields(ds: Dataset):
    # initialize the frequency counter
    counters = defaultdict(Counter)

    _src_vocab, _src_vocab_size = inputter._load_vocab(ds.vocab.source,'src', counters)
    _tgt_vocab, _tgt_vocab_size = inputter._load_vocab(ds.vocab.target, 'tgt', counters)

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
    fields = inputter.get_fields(defaults.vocabulary["data_type"], src_nfeats, tgt_nfeats)

    return inputter._build_fields_vocab(fields, counters, **defaults.vocabulary)
