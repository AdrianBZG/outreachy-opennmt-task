import onmt
import torch

emb_size = 100
d_model = 1
tx_encoder_layers = 1
tx_decoder_layers = 1
lr=1
heads=5


def SimpleTransformer(vocab):
    src_vocab = vocab['src'].base_field.vocab
    tgt_vocab = vocab['tgt'].base_field.vocab

    src_padding = src_vocab.stoi[vocab['src'].base_field.pad_token]
    tgt_padding = tgt_vocab.stoi[vocab['tgt'].base_field.pad_token]

    encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab), word_padding_idx=src_padding)
    encoder = onmt.encoders.transformer.TransformerEncoder(
        num_layers=tx_encoder_layers, d_model=d_model,
        embeddings=encoder_embeddings,
        heads=heads
    )

        # d_ff (int): size of the inner FF layer
        # dropout (float): dropout in residual, self-attn(dot) and feed-forward
        # attention_dropout (float): dropout in context_attn (and self-attn(avg))
    
    decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab), word_padding_idx=tgt_padding)
    decoder = onmt.decoders.transformer.TransformerDecoder(
        num_layers=tx_decoder_layers, d_model=d_model,
        embeddings=decoder_embeddings,
        heads=heads,
        self_attn_type="average"
    )

    model = onmt.models.model.NMTModel(encoder, decoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Specify the tgt word generator and loss computation module
    # model.generator = torch.nn.Sequential(
    #     torch.nn.Linear(rnn_size, len(tgt_vocab)),
    #     torch.nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=torch.nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, max_grad_norm=2)

    return model, loss, optimizer