from argparse import ArgumentParser
from config import defaults

sys_args = [
    # ('--dataset', dest='dataset', type=str, help='Name of the candidate')
    # ('--surname', dest='surname', type=str, help='Surname of the candidate')
    # ('--age', dest='age', type=int, help='Age of the candidate')
]

class Parser():
    def __init__():
        pass

def arg_parser():
    parser = ArgumentParser(description='Application Parameters')
    for arg in sys_args:
        parser.add_argument(arg)

    return parser.parse_args()


def setup_config_args(args):
    for key in args:
        defaults.datasets.append(key)