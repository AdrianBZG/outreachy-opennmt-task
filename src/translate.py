from onmt import inputters
from onmt.translate import Translator, TranslationBuilder
from torch import cuda

def translate(model, vocab):
    _readers, _data = inputters.Dataset.config([
        ("src", { "reader": inputters.str2reader["text"](),  "data": "ds.val.source" }), 
        ("tgt", { "reader": inputters.str2reader["text"](), "data": "ds.val.target" })
    ])

    dataset = inputters.Dataset(vocab, _readers, _data, sort_key=inputters.str2sortkey["text"])
    data_iter = inputters.OrderedIterator(
        dataset=dataset,
        batch_size=10,
        train=False,
        sort=False,
        sort_within_batch=True,
        shuffle=False
    )

    builder = TranslationBuilder(data=dataset, fields=vocab)

    src_reader = inputters.str2reader["text"]
    tgt_reader = inputters.str2reader["text"]
    gpu = 0 if cuda.is_available() else -1

    translator = Translator(model=model, 
        fields=vocab, 
        src_reader=src_reader, 
        tgt_reader=tgt_reader, 
        gpu=gpu
    )

    trans_batch = translator.translate_batch(data_iter, [vocab["src"].base_field.vocab], attn_debug=False)
    translations = builder.from_batch(trans_batch)

    return translations
