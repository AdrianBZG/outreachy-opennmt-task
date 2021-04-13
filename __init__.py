import sys
from config import genconfig
from src.preprocess import setup_dataset

print(f"Name of the script      : {sys.argv[0]}")
print(f"Arguments of the script : {sys.argv[1:]}")

# app_args = genconfig.arg_parser()
# genconfig.setup_config_args('abcd')

setup_dataset('toy-ende')