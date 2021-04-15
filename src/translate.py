import onmt

src_data = {"reader": onmt.inputters.str2reader["text"](), "data": src_val}
tgt_data = {"reader": onmt.inputters.str2reader["text"](), "data": tgt_val}
_readers, _data = onmt.inputters.Dataset.config(
    [('src', src_data), ('tgt', tgt_data)])

dataset = onmt.inputters.Dataset(
    vocab_fields, readers=_readers, data=_data,
    sort_key=onmt.inputters.str2sortkey["text"])

data_iter = onmt.inputters.OrderedIterator(
            dataset=dataset,
            device="cuda",
            batch_size=10,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = GNMTGlobalScorer(alpha=0.7, 
                          beta=0., 
                          length_penalty="avg", 
                          coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = Translator(model=model, 
                        fields=vocab_fields, 
                        src_reader=src_reader, 
                        tgt_reader=tgt_reader, 
                        global_scorer=scorer,
                        gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=dataset, 
                                            fields=vocab_fields)


for batch in data_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=[src_vocab],
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break