from abc import ABC, abstractmethod
from src.utils.dataset import Dataset

class DataReader(ABC):
    """
    Base class to implement a reader for a dataset
    """

    dataname = str()
    datapath = str()

    def __init__(self, datapath: str) -> None:
        """
        Provide the path of the folder of the dataset
        """
        super().__init__()
        self.datapath = datapath

    @abstractmethod
    def _parse_data(self, trainsplit: float, holdout: float) -> Dataset:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'<DataReader: {self.dataname}>'
