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
    path = ".".join(path)
    if name == "trainsplit": 
        global trainsplit
        trainsplit = type(trainsplit)(value)
    elif name == "holdout": 
        global holdout
        holdout = type(holdout)(value)
    elif name == "tokenizer": 
        global tokenizer
        if path == "symbols": 
            tokenizer["symbols"] = type(tokenizer["symbols"])(value)
        elif path == "args.mode": 
            tokenizer["args"]["mode"] = type(tokenizer["args"]["mode"])(value)
        elif path == "args.joiner_annotate": 
            tokenizer["args"]["joiner_annotate"] = type(tokenizer["args"]["joiner_annotate"])(value)
        elif path == "args.preserve_placeholders": 
            tokenizer["args"]["preserve_placeholders"] = type(tokenizer["args"]["preserve_placeholders"])(value)
        elif path == "args.case_markup": 
            tokenizer["args"]["case_markup"] = type(tokenizer["args"]["case_markup"])(value)
        elif path == "args.soft_case_regions": 
            tokenizer["args"]["soft_case_regions"] = type(tokenizer["args"]["soft_case_regions"])(value)
        elif path == "args.preserve_segmented_tokens": 
            tokenizer["args"]["preserve_segmented_tokens"] = type(tokenizer["args"]["preserve_segmented_tokens"])(value)
    elif name == "vocabulary": 
        global vocabulary
        if path == "data_type": 
            vocabulary["data_type"] = type(vocabulary["data_type"])(value)
        elif path == "share_vocab": 
            vocabulary["share_vocab"] = type(vocabulary["share_vocab"])(value)
        elif path == "vocab_size_multiple": 
            vocabulary["vocab_size_multiple"] = type(vocabulary["vocab_size_multiple"])(value)
        elif path == "src_vocab_size": 
            vocabulary["src_vocab_size"] = type(vocabulary["src_vocab_size"])(value)
        elif path == "tgt_vocab_size": 
            vocabulary["tgt_vocab_size"] = type(vocabulary["tgt_vocab_size"])(value)
        elif path == "src_words_min_frequency": 
            vocabulary["src_words_min_frequency"] = type(vocabulary["src_words_min_frequency"])(value)
        elif path == "tgt_words_min_frequency": 
            vocabulary["tgt_words_min_frequency"] = type(vocabulary["tgt_words_min_frequency"])(value)
    elif name == "dropout": 
        global dropout
        dropout = type(dropout)(value)
    elif name == "training": 
        global training
        if path == "train_steps": 
            training["train_steps"] = type(training["train_steps"])(value)
        elif path == "valid_steps": 
            training["valid_steps"] = type(training["valid_steps"])(value)
        elif path == "save_checkpoint_steps": 
            training["save_checkpoint_steps"] = type(training["save_checkpoint_steps"])(value)
    elif name == "lstm": 
        global lstm
        if path == "emb_size":
            lstm["emb_size"] = type(lstm["emb_size"])(value)
        elif path == "learning_rate":
            lstm["learning_rate"] = type(lstm["learning_rate"])(value)
        elif path == "encoder.rnn_type": 
            lstm["encoder"]["rnn_type"] = type(lstm["encoder"]["rnn_type"])(value)
        elif path == "encoder.hidden_size": 
            lstm["encoder"]["hidden_size"] = type(lstm["encoder"]["hidden_size"])(value)
        elif path == "encoder.num_layers": 
            lstm["encoder"]["num_layers"] = type(lstm["encoder"]["num_layers"])(value)
        elif path == "encoder.bidirectional": 
            lstm["encoder"]["bidirectional"] = type(lstm["encoder"]["bidirectional"])(value)
        elif path == "decoder.rnn_type": 
            lstm["decoder"]["rnn_type"]  = type(lstm["decoder"]["rnn_type"])(value)
        elif path == "decoder.hidden_size": 
            lstm["decoder"]["hidden_size"]  = type(lstm["decoder"]["hidden_size"])(value)
        elif path == "decoder.num_layers": 
            lstm["decoder"]["num_layers"] = type(lstm["decoder"]["num_layers"])(value)
        elif path == "decoder.bidirectional_encoder": 
            lstm["decoder"]["bidirectional_encoder"] = type(lstm["decoder"]["bidirectional_encoder"])(value)
    elif name == "transformer":
        global transformer
        if path == "emb_size":
            transformer["emb_size"] = type(transformer["emb_size"])(value)
        elif path == "learning_rate":
            transformer["learning_rate"] = type(transformer["learning_rate"])(value)
        elif path == "encoder.d_model":
            transformer["d_model"] = type(transformer["d_model"])(value)
        elif path == "encoder.num_layers":
            transformer["num_layers"] = type(transformer["num_layers"])(value)
        elif path == "encoder.heads":
            transformer["heads"] = type(transformer["heads"])(value)
        elif path == "encoder.d_ff":
            transformer["d_ff"] = type(transformer["d_ff"])(value)
        elif path == "encoder.dropout":
            transformer["dropout"] = type(transformer["dropout"])(value)
        elif path == "encoder.attention_dropout":
            transformer["attention_dropout"] = type(transformer["attention_dropout"])(value)
        elif path == "encoder.max_relative_positions":
            transformer["max_relative_positions"] = type(transformer["max_relative_positions"])(value)
        elif path == "decoder.d_model":
            transformer["d_model"] = type(transformer["d_model"])(value)
        elif path == "decoder.num_layers":
            transformer["num_layers"] = type(transformer["num_layers"])(value)
        elif path == "decoder.heads":
            transformer["heads"] = type(transformer["heads"])(value)
        elif path == "decoder.self_attn_type":
            transformer["self_attn_type"] = type(transformer["self_attn_type"])(value)
        elif path == "decoder.d_ff":
            transformer["d_ff"] = type(transformer["d_ff"])(value),
        elif path == "decoder.dropout":
            transformer["dropout"] = type(transformer["dropout"])(value),
        elif path == "decoder.attention_dropout":
            transformer["attention_dropout"] = type(transformer["attention_dropout"])(value),
        elif path == "decoder.copy_attn":
            transformer["copy_attn"] = type(transformer["copy_attn"])(value)
        elif path == "decoder.max_relative_positions":
            transformer["max_relative_positions"] = type(transformer["max_relative_positions"])(value)
        elif path == "decoder.aan_useffn":
            transformer["aan_useffn"] = type(transformer["aan_useffn"])(value)
        elif path == "decoder.full_context_alignment":
            transformer["full_context_alignment"] = type(transformer["full_context_alignment"])(value)
        elif path == "decoder.alignment_layer":
            transformer["alignment_layer"] = type(transformer["alignment_layer"])(value)
        elif path == "decoder.alignment_heads":
            transformer["alignment_heads"] = type(transformer["alignment_heads"])(value)
        