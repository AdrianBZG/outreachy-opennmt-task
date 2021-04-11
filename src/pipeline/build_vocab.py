from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts

parser = ArgumentParser()
dynamic_prepare_opts(parser, build_vocab_only=True)