import onmt
import torch

emb_size = 100
rnn_size = 500
rnn_encoder_layers = 1
rnn_decoder_layers = 1
lr = 1
# Specify the core model.


def BaseLSTMModel(vocab):
    src_vocab = vocab['src'].base_field.vocab
    tgt_vocab = vocab['tgt'].base_field.vocab

    src_padding = src_vocab.stoi[vocab['src'].base_field.pad_token]
    tgt_padding = tgt_vocab.stoi[vocab['tgt'].base_field.pad_token]
        
    encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab), word_padding_idx=src_padding)
    encoder = onmt.encoders.RNNEncoder(
        hidden_size=rnn_size, num_layers=rnn_encoder_layers,
        bidirectional=True, rnn_type="LSTM", 
        embeddings=encoder_embeddings
    )

    decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab), word_padding_idx=tgt_padding)
    decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
        hidden_size=rnn_size, num_layers=rnn_decoder_layers, 
        bidirectional_encoder=True, rnn_type="LSTM", 
        embeddings=decoder_embeddings
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = onmt.models.model.NMTModel(encoder, decoder)
    model.to(device)

    # Specify the tgt word generator and loss computation module
    model.generator = torch.nn.Sequential(
        torch.nn.Linear(rnn_size, len(tgt_vocab)),
        torch.nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=torch.nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, max_grad_norm=2)

    return model, loss, optimizer
