from os import path
from collections import defaultdict, Counter

from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.bin.build_vocab import build_vocab_main
from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields

from src.utils.dataset import Dataset


def _build_vocabulary(ds: Dataset) -> None:
    base_args = ([
        "-config", f"{path.join(ds.path, 'config.yaml')}", 
        "-n_sample", "10000"
    ])

    parser = ArgumentParser(description='vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only=True)

    options, unknown = parser.parse_known_args(base_args)
    build_vocab_main(options)
    
    return options, unknown


def _build_vocab(ds: Dataset):
    # initialize the frequency counter
    counters = defaultdict(Counter)

    _src_vocab, _src_vocab_size = _load_vocab(ds.vocab.source,'src', counters)
    _tgt_vocab, _tgt_vocab_size = _load_vocab(ds.vocab.target, 'tgt', counters)

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
    fields = get_fields('text', src_nfeats, tgt_nfeats)

    # build fields vocab config
    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = 30000
    tgt_vocab_size = 30000
    src_words_min_frequency = 1
    tgt_words_min_frequency = 1

    vocab_fields = _build_fields_vocab(
        fields, counters, 'text', share_vocab,
        vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency)

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]