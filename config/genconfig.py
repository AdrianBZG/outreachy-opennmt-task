import yaml
from os import path
from config import defaults
from src.utils.dataset import DataItem, Dataset

yamlconf = lambda ds: f"""## Where the samples will be written
save_data: {path.join(ds.path, 'run', 'samples')}

## Where the vocab(s) will be written
src_vocab: {ds.vocab.source}
tgt_vocab: {ds.vocab.target}

# Prevent overwriting existing files in the folder
overwrite: False

# Corpus opts:
data:
    corpus:
        path_src: {ds.train.source}
        path_tgt: {ds.train.target}
    valid:
        path_src: {ds.val.source}
        path_tgt: {ds.val.target}

# Train on a single GPU
world_size: 1
gpu_ranks: [0]

# Where to save the checkpoints
save_model: {path.join(ds.path, 'run', 'model')}
save_checkpoint_steps: 500
train_steps: 1000
valid_steps: 500
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