import yaml
import json

base_args = (["-config", "toy-ende/config.yaml", "-n_sample", "10000"])

datasets = ['toy-ende', 'rapid2016']
datapaths = {
    "toy-ende": "./data/toy-ende",
    "rapid2016": "./data/rapid2016"
}

trainsplit = 0.6
holdout = 0.2

tokenizer = {
    "mode": "aggressive",
    "joiner_annotate": True,
    "preserve_placeholders": True,
    "case_markup": True,
    "soft_case_regions": True,
    "preserve_segmented_tokens": True,
}

yaml_config = """
## Where the vocab(s) will be written
save_data: toy-ende/run/example
src_vocab: toy-ende/run/example.vocab.src
tgt_vocab: toy-ende/run/example.vocab.tgt
# Corpus opts:
data:
    corpus:
        path_src: toy-ende/src-train.txt
        path_tgt: toy-ende/tgt-train.txt
        transforms: []
        weight: 1
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
        transforms: []
"""
config = yaml.safe_load(yaml_config)