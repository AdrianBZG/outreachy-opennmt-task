class Dataset():
    """
    Standard dataset class
    """

    name = str()

    train = []
    test = []
    val = []

    def __init__(self, name: str = 'nodata') -> None:
        self.name = name
        pass

    def __repr__(self) -> str:
        return f'<Dataset: {self.name}>'
