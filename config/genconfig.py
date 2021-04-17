import yaml
from os import path
from config import defaults
from src.utils.dataset import DataItem, Dataset

yamlconf = lambda ds: f"""## Where the samples will be written
save_data: {path.join(ds.path, 'samples')}

# Where to save the checkpoints
save_model: {path.join('trained', ds.name, 'model')}

## Where the vocab(s) will be written
src_vocab: {ds.vocab.source}
tgt_vocab: {ds.vocab.target}

# Overwriting existing files in the folder
overwrite: True

# Corpus opts:
data:
    corpus:
        path_src: {ds.train.source}
        path_tgt: {ds.train.target}
        # transforms: [sentencepiece, filtertoolong]
    valid:
        path_src: {ds.val.source}
        path_tgt: {ds.val.target}
        # transforms: [sentencepiece, filtertoolong]

# Train on a single GPU
# world_size: 1
# gpu_ranks: [0]

# ### Transform related opts:
# #### Subword
src_subword_type: bpe
src_subword_model: {path.join('data', ds.name, 'run', 'subwords.bpe')}
src_onmttok_kwargs: {str(defaults.tokenizer_args)}
tgt_onmttok_kwargs: {str(defaults.tokenizer_args)}
transforms: [onmt_tokenize]

# src_subword_nbest: 1
# src_subword_alpha: 0.0
# tgt_subword_nbest: 1
# tgt_subword_alpha: 0.0

# #### Filter
# src_seq_length: 150
# tgt_seq_length: 150
"""


def gen_yaml_config(ds: Dataset):
    ds.vocab = DataItem(
        f"{path.join(ds.path, 'run', ds.name)}.vocab.src",
        f"{path.join(ds.path, 'run', ds.name)}.vocab.tgt"
    )

    conf = yamlconf(ds)
    options = yaml.safe_load(conf)
    with open(f"{path.join(ds.path, 'config.yaml')}", "w") as f:
        f.write(conf)      

    return options