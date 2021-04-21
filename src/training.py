from torch import cuda
from onmt import Trainer
from onmt.utils import ReportMgr
from onmt.inputters import corpus, inputter, dynamic_iterator
from config import defaults

from onmt.models.model import NMTModel
from onmt.utils.loss import NMTLossCompute
from onmt.utils.optimizers import Optimizer
from src.utils.dataset import Dataset


def training_iterator(ds: Dataset, vocab):
    # build the ParallelCorpus
    train = corpus.ParallelCorpus("corpus", ds.train.source, ds.train.target)
    valid = corpus.ParallelCorpus("valid", ds.val.source, ds.val.target)

    # build the training iterator
    iterator = dynamic_iterator.DynamicDatasetIter(
        corpora={"corpus": train},
        corpora_info={"corpus": {"weight": 1}},
        transforms={},
        fields=vocab,
        is_train=True,
        batch_type="tokens",
        batch_size=4096,
        batch_size_multiple=1,
        data_type="text")

    # build the validation iterator
    validator = dynamic_iterator.DynamicDatasetIter(
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
    iterator = iter(inputter.IterOnDevice(iterator, 0 if cuda.is_available() else -1))
    validator = iter(inputter.IterOnDevice(validator, 0 if cuda.is_available() else -1))

    return iterator, validator


def training_session(model: NMTModel, loss: NMTLossCompute, opt: Optimizer, dropout: float = defaults.dropout):
    report_manager = ReportMgr(report_every=50, start_time=None, tensorboard_writer=None)
    session = Trainer(
        model=model,
        train_loss=loss,
        valid_loss=loss,
        optim=opt,
        report_manager=report_manager,
        dropout=dropout)

    return session