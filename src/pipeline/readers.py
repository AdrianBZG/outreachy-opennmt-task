from os import path
from typing import List
from config import defaults
from src.utils.dataset import DataItem, Dataset
from src.utils.preprocessing import DataReader


class RapidSETReader(DataReader):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        self.dataname = 'rapid2016'

    def _parse_data(self, trainsplit: float = defaults.trainsplit, holdout: float = defaults.holdout) -> Dataset:
        print(f'Reading {self.dataname} from {self.datapath}')

        rapid = Dataset(name=self.dataname, path=self.datapath)
        return rapid


class WikiTitlesReader(DataReader):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        raise NotImplementedError


class ToyENDEReader(DataReader):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        self.dataname = 'toy-ende'

    def _parse_data(self, trainsplit = None, holdout = None) -> Dataset:
        print(f'Reading {self.dataname} from {self.datapath}')

        if trainsplit is not None:
            raise NotImplementedError('This dataset does not support custom train-test split.')

        if holdout is not None:
            raise NotImplementedError('This dataset does not support custom holdout.')

        toyende = Dataset(name=self.dataname, path=self.datapath)
        toyende.train = DataItem(
            path.join(self.datapath, 'src-train.txt'),
            path.join(self.datapath, 'tgt-train.txt')
        )
        toyende.test = DataItem(
            path.join(self.datapath, 'src-test.txt'),
            path.join(self.datapath, 'tgt-test.txt')
        )
        toyende.val = DataItem(
            path.join(self.datapath, 'src-val.txt'),
            path.join(self.datapath, 'tgt-val.txt')
        )

        return toyende