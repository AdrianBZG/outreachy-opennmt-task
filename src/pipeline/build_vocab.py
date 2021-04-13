from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts

parser = ArgumentParser()
dynamic_prepare_opts(parser, build_vocab_only=True)

src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]