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

    train = DataItem
    test = DataItem
    val = DataItem

    def __init__(self, name: str = 'nodata') -> None:
        self.name = name
        pass

    def __repr__(self) -> str:
        return f'<Dataset {self.name} \n\ttrain: {self.train} \n\ttest: {self.test} \n\tval: {self.val}>'

    def items(self):
        return [self.train, self.test, self.val]
