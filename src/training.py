from typing import Iterator
from onmt import Trainer
from onmt.utils import ReportMgr
from onmt.inputters import inputter, corpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter

from src.utils.dataset import Dataset


def setup_training_iterator(ds: Dataset, vocab):
    # build the ParallelCorpus
    corpus = corpus.ParallelCorpus("corpus", ds.train.source, ds.train.target)
    valid = corpus.ParallelCorpus("valid", ds.val.source, ds.val.target)

    # build the training iterator
    iterator = DynamicDatasetIter(
        corpora={"corpus": corpus},
        corpora_info={"corpus": {"weight": 1}},
        transforms={},
        fields=vocab,
        is_train=True,
        batch_type="tokens",
        batch_size=4096,
        batch_size_multiple=1,
        data_type="text")

    # build the validation iterator
    validator = DynamicDatasetIter(
        corpora={"valid": valid},
        corpora_info={"valid": {"weight": 1}},
        transforms={},
        fields=vocab,
        is_train=False,
        batch_type="sents",
        batch_size=8,
        batch_size_multiple=1,
        data_type="text")
        
    # make sure the iteration happens on GPU 0 (-1 for CPU, N for GPU N)
    # iterator = iter(inputter.IterOnDevice(iterator, 0))
    # validator = iter(inputter.IterOnDevice(validator, 0))

    return iterator, validator


def train(model, train_iter: Iterator, valid_iter: Iterator):
    report_manager = ReportMgr(report_every=50, start_time=None, tensorboard_writer=None)
    trainer = Trainer(model=model,
        train_loss=loss,
        valid_loss=loss,
        optim=optim,
        report_manager=report_manager,
        dropout=[0.1])

    trainer.train(
        train_iter=train_iter, train_steps=1000,
        valid_iter=valid_iter, valid_steps=500)

    return