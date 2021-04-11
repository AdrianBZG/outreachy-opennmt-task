import pyonmttok

args = {
    "mode": "aggressive",
    "joiner_annotate": True,
    "preserve_placeholders": True,
    "case_markup": True,
    "soft_case_regions": True,
    "preserve_segmented_tokens": True,
}

n_symbols = 40000

tokenizer_default = pyonmttok.Tokenizer(**args)
learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=n_symbols)
# load training corpus
learner.ingest_file("wiki.train.raw")

# learn and store bpe model
tokenizer = learner.learn("subwords.bpe")

# tokenize corpus and save results
for data_file in ["wiki.valid", "wiki.test", "wiki.train"]:
    tokenizer.tokenize_file(f"{data_file}.raw", f"{data_file}.bpe")
