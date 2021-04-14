from sys import argv
from onmt.utils.logging import init_logger

from src.preprocess import setup_dataset, setup_vocab

def main():    
    init_logger()
    try:
        ds = setup_dataset('toy-ende')
        setup_vocab(ds)
    except (RuntimeError):
        return RuntimeError
    
    return 0


if __name__ == '__main__':  
    print(f"Name of the script      : {argv[0]}")
    print(f"Arguments of the script : {argv[1:]}")
    print()

    main()