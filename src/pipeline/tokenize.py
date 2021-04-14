from pyonmttok import Tokenizer, BPELearner
from config import defaults
from utils.dataset import Dataset


def tokenize_corpus(dataset: Dataset, outpath: str):
    """
    Tokenize the corpus from the prepared dataset.
    """

    tokenizer_default = Tokenizer(**defaults.tokenizer_args)
    learner = BPELearner(tokenizer=tokenizer_default, symbols=defaults.tokenizer_nsymbols)

    # load training corpus
    learner.ingest_file("wiki.train.raw")

    # learn and store bpe model
    tokenizer = learner.learn("subwords.bpe")

    # tokenize corpus and save results
    for data_file in ["wiki.valid", "wiki.test", "wiki.train"]:
        tokenizer.tokenize_file(f"{data_file}", f"{data_file}")
