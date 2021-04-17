from os import path
from typing import List
from config import defaults
from src.utils.dataset import DataItem, Dataset
from src.utils.preprocessing import DataReader

import pyonmttok

class RapidSETReader(DataReader):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        self.dataname = 'rapid2016'

    def _parse_data(self, tokenize: bool = False, trainsplit: float = defaults.trainsplit, holdout: float = defaults.holdout) -> Dataset:
        print(f'Reading {self.dataname} from {self.datapath}')

        if trainsplit >= 1 or holdout >= 1:
            raise ValueError('trainsplit and holdout must be < 1')

        # Split into train, test, val

        rapid = Dataset(name=self.dataname, path=self.datapath)
        rapid.train = DataItem(
            path.join(self.datapath, 'rapid2016.de-en.de'),
            path.join(self.datapath, 'rapid2016.de-en.en')
        )

        if tokenize:
            self.tokenize()

        return rapid

    def tokenize(self):
        print('Tokenizing and training BPE model')
        
        tokenizer_default = pyonmttok.Tokenizer(**defaults.tokenizer_args)
        learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=defaults.tokenizer_nsymbols)

        # load training corpus
        learner.ingest_file(path.join('data', 'rapid2016', 'rapid2016.de-en.de'))

        # learn and store bpe model
        tokenizer = learner.learn(path.join('data', 'rapid2016', 'run', 'subwords.bpe'))

        # tokenize corpus and save results
        # for data_file in ['src-train', 'src-test', 'src-val']:
        #     data_file = path.join('data', 'toy-ende', data_file) 
        #     tokenizer.tokenize_file(f'{data_file}.txt', f'{data_file}.bpe')
        
        return


class WikiTitlesReader(DataReader):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        self.dataname = 'wiki'
        raise NotImplementedError


class ToyENDEReader(DataReader):
    def __init__(self, datapath: str) -> None:
        super().__init__(datapath)
        self.dataname = 'toy-ende'

    def _parse_data(self, tokenize: bool = True, trainsplit = None, holdout = None) -> Dataset:
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

        if tokenize:
            self.tokenize()

        return toyende

    def tokenize(self):
        print('Tokenizing and training BPE model')
        
        tokenizer_default = pyonmttok.Tokenizer(**defaults.tokenizer_args)
        learner = pyonmttok.BPELearner(tokenizer=tokenizer_default, symbols=defaults.tokenizer_nsymbols)

        # load training corpus
        learner.ingest_file(path.join('data', 'toy-ende', 'src-train.txt'))

        # learn and store bpe model
        tokenizer = learner.learn(path.join('data', 'toy-ende', 'run', 'subwords.bpe'))

        # tokenize corpus and save results
        for data_file in ['src-train', 'src-test', 'src-val']:
            data_file = path.join('data', 'toy-ende', data_file) 
            tokenizer.tokenize_file(f'{data_file}.txt', f'{data_file}.bpe')
        
        return