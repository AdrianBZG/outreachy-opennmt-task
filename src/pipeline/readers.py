from typing import List
from config import defaults
from src.utils.dataset import Dataset
from src.utils.preprocessing import DataReader

class ToyENDEReader(DataReader):
    def __init__(self, datapath: str) -> None:
        self.dataname = 'toy-ende'
        super().__init__(defaults.datapaths[self.dataname])

    def _parse_data(self, trainsplit = None, holdout = None) -> Dataset:
        print(f'Reading {self.dataname} from {self.datapath}')

        if trainsplit is not None:
            raise NotImplementedError('This dataset does not support custom train-test split.')

        if holdout is not None:
            raise NotImplementedError('This dataset does not support custom holdout.')

        return Dataset(self.dataname)


class RapidSETRead(DataReader):
    def __init__(self, datapath: str) -> None:
        self.dataname = 'rapid2016'
        super().__init__(defaults.datapaths[self.dataname])

    def _parse_data(self, trainsplit = defaults.trainsplit, holdout = defaults.holdout) -> Dataset:
        print(f'Reading {self.dataname} from {self.datapath}')
        return Dataset(self.dataname)