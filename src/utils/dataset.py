from dataclasses import dataclass

@dataclass
class DataItem():
    """
    Simple class to represent a tuple of src-target files in a parallel corpus.
    """
    source: str
    target: str

    def __init__(self, src: str, tgt: str) -> None:
        self.source = src
        self.target = tgt

    def __repr__(self) -> str:
        return f'(src: {self.source} tgt: {self.target})'


class Dataset():
    """
    Standard dataset class with train, test and validation splits.
    """

    name = str
    path = str

    vocab = DataItem

    train = DataItem
    val = DataItem
    test = DataItem

    def __init__(self, name: str = 'nodata', path: str = './') -> None:
        self.name = name
        self.path = path
        pass

    def __repr__(self) -> str:
        return f'<Dataset {self.name} {self.path} \n\ttrain: {self.train} \n\tval: {self.val} \n\ttest: {self.test} >'

    def items(self):
        return [self.train, self.test, self.val]
