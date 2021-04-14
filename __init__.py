import sys
from config import genconfig
from src.preprocess import setup_dataset

def main():
    print(f"Name of the script      : {sys.argv[0]}")
    print(f"Arguments of the script : {sys.argv[1:]}")

    ds = setup_dataset('toy-ende')
    print(ds)


if __name__ == '__main__':  
    main()