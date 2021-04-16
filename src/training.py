from torch import cuda
from typing import Iterator, List
from onmt import Trainer
from onmt.utils import ReportMgr
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.inputter import IterOnDevice
from onmt.inputters.dynamic_iterator import DynamicDatasetIter

from src.utils.dataset import Dataset
from onmt.models.model import NMTModel
from onmt.utils.loss import NMTLossCompute
from onmt.utils.optimizers import Optimizer


def training_iterator(ds: Dataset, vocab):
    # build the ParallelCorpus
    corpus = ParallelCorpus("corpus", ds.train.source, ds.train.target)
    valid = ParallelCorpus("valid", ds.val.source, ds.val.target)

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
    iterator = iter(IterOnDevice(iterator, 0 if cuda.is_available() else -1))
    validator = iter(IterOnDevice(validator, 0 if cuda.is_available() else -1))

    return iterator, validator


def training_session(model: NMTModel, loss: NMTLossCompute, opt: Optimizer, dropout: List[float] = [0.1]):
    report_manager = ReportMgr(report_every=50, start_time=None, tensorboard_writer=None)
    session = Trainer(
        model=model,
        train_loss=loss,
        valid_loss=loss,
        optim=opt,
        report_manager=report_manager,
        dropout=dropout)

    return session