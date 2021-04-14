import sys
from config import genconfig
from onmt.utils.logging import init_logger

from src.preprocess import setup_dataset, setup_vocab

def main():    
    init_logger()

    print(f"Name of the script      : {sys.argv[0]}")
    print(f"Arguments of the script : {sys.argv[1:]}")

    ds = setup_dataset('toy-ende')
    setup_vocab(ds)


if __name__ == '__main__':  
    main()