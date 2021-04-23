datasets = ["toy-ende", "rapid2016", "wiki"]
datapaths = {
    "toy-ende": "data/toy-ende",
    "rapid2016": "data/rapid2016",
    "wiki": "data/wiki"
}

trainsplit = 0.6
holdout = 0.2

# Default config options for pyonmttok.Tokenizer
tokenizer = {
    "symbols": 40000,
    "args": {
        "mode": "aggressive",
        "joiner_annotate": True,
        "preserve_placeholders": True,
        "case_markup": True,
        "soft_case_regions": True,
        "preserve_segmented_tokens": True
    }
}

# Default config options for onmt.inputters.inputter._build_fields_vocab
vocabulary = {
    "data_type": "text",
    "share_vocab": False,
    "vocab_size_multiple": 1,
    "src_vocab_size": 30000,
    "tgt_vocab_size": 30000,
    "src_words_min_frequency": 1,
    "tgt_words_min_frequency": 1
}

# Default config options for onmt.Trainer
dropout = 0.1
training = {
    "train_steps": 100000,
    "valid_steps": 4000,
    "save_checkpoint_steps": 4000
}

# Default config options for model.lstm.BaseLSTMModel
lstm = {
    "emb_size": 100,
    "learning_rate": 1,
    "encoder": {
        "rnn_type": "LSTM", 
        "hidden_size": 500, 
        "num_layers": 1,
        "bidirectional": True
    },
    "decoder": {
        "rnn_type": "LSTM", 
        "hidden_size": 500, 
        "num_layers": 1,
        "bidirectional_encoder": True
    }
}

# Default config options for model.transformer.SimpleTransformer
transformer = {
    "emb_size": 512,
    "learning_rate": 2,
    "encoder" : {
        "d_model": 512,
        "num_layers": 6,
        "heads": 8,
        "d_ff": 2048, 
        "dropout": 0.1, 
        "attention_dropout": 0.1,
        "max_relative_positions": 0
    },
    "decoder" : {
        "d_model": 512,
        "num_layers": 6,
        "heads": 8,
        "self_attn_type": "average",
        "d_ff": 2048, # size of the inner FF layer
        "dropout": 0.1, # dropout in residual, self-attn(dot) and feed-forward
        "attention_dropout": 0.1, # attention_dropout (float): dropout in context_attn (and self-attn(avg))
        "copy_attn": True,
        "max_relative_positions": 0,
        "aan_useffn": False,
        "full_context_alignment": False,
        "alignment_layer": 1,
        "alignment_heads": 0
    }
}

def bindings(name, path, value):
    if len(path) == 1: 
        primary = path[0]
    elif len(path) == 2: 
        primary, secondary = path
    elif len(path) > 2:
        raise KeyError('Such deep nesting is not supported')

    if name == "trainsplit": 
        global trainsplit
        trainsplit = type(trainsplit)(value)
    elif name == "holdout": 
        global holdout
        holdout = type(holdout)(value)
    elif name == "tokenizer": 
        global tokenizer
        if len(path) == 1: 
            tokenizer[primary] = type(tokenizer[primary])(value)
        elif len(path) == 2: 
            tokenizer[primary][secondary] = type(tokenizer[primary][secondary])(value)
    elif name == "vocabulary": 
        global vocabulary
        if len(path) == 1: 
            vocabulary[primary] = type(vocabulary[primary])(value)
        elif len(path) == 2: 
            vocabulary[primary][secondary] = type(vocabulary[primary][secondary])(value)
    elif name == "dropout": 
        global dropout
        dropout = type(dropout)(value)
    elif name == "training": 
        global training
        if len(path) == 1: 
            training[primary] = type(training[primary])(value)
        elif len(path) == 2: 
            training[primary][secondary] = type(training[primary][secondary])(value)
    elif name == "lstm": 
        global lstm
        if len(path) == 1: 
            lstm[primary] = type(lstm[primary])(value)
        elif len(path) == 2: 
            lstm[primary][secondary] = type(lstm[primary][secondary])(value)
    elif name == "transformer":
        global transformer
        if len(path) == 1: 
            transformer[primary] = type(transformer[primary])(value)
        elif len(path) == 2: 
            transformer[primary][secondary] = type(transformer[primary][secondary])(value)
