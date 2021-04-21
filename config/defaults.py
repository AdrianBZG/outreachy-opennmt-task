datasets = ['toy-ende', 'rapid2016', 'wiki']
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
    "train_steps": 500,
    "valid_steps": 200,
    "save_checkpoint_steps": 50
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
    "emb_size": 100,
    "learning_rate": 1,
    "encoder" : {
        "num_layers": 6,
        "d_model": 5,
        "heads": 8
    },
    "decoder" : {
        "num_layers": 6,
        "d_model": 5,
        "heads": 8,
        "self_attn_type": "average",
        # d_ff (int): size of the inner FF layer
        # dropout (float): dropout in residual, self-attn(dot) and feed-forward
        # attention_dropout (float): dropout in context_attn (and self-attn(avg))
    }
}