datasets = ['toy-ende', 'rapid2016', 'wiki']
datapaths = {
    "toy-ende": "data/toy-ende",
    "rapid2016": "data/rapid2016",
    "wiki": "data/wiki"
}

trainsplit = 0.6
holdout = 0.2

tokenizer_nsymbols = 40000
tokenizer_args = {
    "mode": "aggressive",
    "joiner_annotate": True,
    "preserve_placeholders": True,
    "case_markup": True,
    "soft_case_regions": True,
    "preserve_segmented_tokens": True,
}