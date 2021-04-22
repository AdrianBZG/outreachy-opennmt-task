from onmt import inputters
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from torch import cuda
from src.utils.dataset import Dataset

def evaluation(model, ds: Dataset, vocab):
    _readers, _data = inputters.Dataset.config([
        ("src", { "reader": inputters.str2reader["text"](),  "data": ds.val.source }), 
        ("tgt", { "reader": inputters.str2reader["text"](), "data": ds.val.target })
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

    scorer = GNMTGlobalScorer(alpha=0.7, beta=0., length_penalty="avg", coverage_penalty="none")
    builder = TranslationBuilder(data=dataset, fields=vocab)

    src_reader = inputters.str2reader["text"]
    tgt_reader = inputters.str2reader["text"]
    gpu = 0 if cuda.is_available() else -1

    translator = Translator(model=model, 
                            fields=vocab, 
                            src_reader=src_reader, 
                            tgt_reader=tgt_reader, 
                            global_scorer=scorer,
                            gpu=gpu)

    for batch in data_iter:
        trans_batch = translator.translate_batch(batch, [vocab["src"].base_field.vocab], attn_debug=False)
        translations = builder.from_batch(trans_batch)

        for trans in translations:
            print(trans.log(0))
        break

    return
