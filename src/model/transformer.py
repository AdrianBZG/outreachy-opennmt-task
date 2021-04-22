from config import defaults
import onmt
import torch


def SimpleTransformer(vocab):
    src_vocab = vocab['src'].base_field.vocab
    tgt_vocab = vocab['tgt'].base_field.vocab

    src_padding = src_vocab.stoi[vocab['src'].base_field.pad_token]
    tgt_padding = tgt_vocab.stoi[vocab['tgt'].base_field.pad_token]

    emb_size = defaults.transformer["emb_size"]

    encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab), word_padding_idx=src_padding)
    encoder = onmt.encoders.transformer.TransformerEncoder(
        embeddings=encoder_embeddings, **defaults.transformer["encoder"])
    
    decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab), word_padding_idx=tgt_padding)
    decoder = onmt.decoders.transformer.TransformerDecoder(
        embeddings=decoder_embeddings, **defaults.transformer["decoder"])

    model = onmt.models.model.NMTModel(encoder, decoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Specify the tgt word generator
    model.generator = torch.nn.Sequential(
        torch.nn.Linear(defaults.transformer["decoder"]["d_model"], len(tgt_vocab)),
        torch.nn.LogSoftmax(dim=-1)
    ).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=torch.nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    lr = defaults.transformer["learning_rate"]

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, max_grad_norm=2)

    return model, loss, optimizer