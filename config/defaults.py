datasets = ['toy-ende', 'rapid2016', 'wiki']
datapaths = {
    "toy-ende": "data/toy-ende",
    "rapid2016": "data/rapid2016",
    "wiki": "data/wiki"
}

trainsplit = 0.6
holdout = 0.2

tokenizer = {
    "nsymbols": 40000,
    "args": {
        "mode": "aggressive",
        "joiner_annotate": True,
        "preserve_placeholders": True,
        "case_markup": True,
        "soft_case_regions": True,
        "preserve_segmented_tokens": True
    }
}

lstm = {
    "emb_size": 100,
    "learning_rate": 1,
    "encoder": {
        "rnn_type": "LSTM", 
        "hidden_size": 500, 
        "num_layers": 1,
        "bidirectional": True
    },
    "encoder": {
        "rnn_type": "LSTM", 
        "hidden_size": 500, 
        "num_layers": 1,
        "bidirectional": True
    }
}

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
        
        # dropout (float): dropout in residual, self-attn(dot) and feed-forward
        # d_ff (int): size of the inner FF layer
        # attention_dropout (float): dropout in context_attn (and self-attn(avg))
    }
}