from sys import argv
from torch import cuda
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed

from src import preprocess, training, evaluate

def main():    
    init_logger()
    is_cuda = cuda.is_available()
    set_random_seed(1111, is_cuda)

    try:
        data = preprocess.setup_dataset('toy-ende')
        vocab, padding = preprocess.setup_vocab(data)

        srcpad, tgtpad = padding

        # Init model
        Model = ''  

        trainer, validator = training.setup_training_iterator(data, vocab)
        report = training.train(Model, trainer, validator)

        evaluate.evaluation(report)
    except (RuntimeError):
        return RuntimeError
    
    return 0


if __name__ == '__main__':  
    print(f"Name of the script      : {argv[0]}")
    print(f"Arguments of the script : {argv[1:]}")
    print()

    exit_code = main()