from pyonmttok import Tokenizer, BPELearner
from config import defaults

n_symbols = 40000

"""
dsfsdf
"""
def tokenize_corpus(path):
    tokenizer_default = Tokenizer(**defaults.tokenizer)
    learner = BPELearner(tokenizer=tokenizer_default, symbols=n_symbols)

    # load training corpus
    learner.ingest_file("wiki.train.raw")

    # learn and store bpe model
    tokenizer = learner.learn("subwords.bpe")

    # tokenize corpus and save results
    for data_file in ["wiki.valid", "wiki.test", "wiki.train"]:
        tokenizer.tokenize_file(f"{data_file}", f"{data_file}")
