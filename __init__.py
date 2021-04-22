from sys import argv
from torch import cuda
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed

from config import defaults
from src import preprocess, training, evaluate
from src.model import lstm, transformer


def main():    
    init_logger()
    is_cuda = cuda.is_available()
    set_random_seed(1111, is_cuda)

    data = preprocess.setup_dataset('toy-ende')
    vocab = preprocess.setup_vocab(data)

    # Init model
    model, loss, opt = transformer.SimpleTransformer(vocab)

    train, validate = training.training_iterator(data, vocab)
    TrainingSession = training.training_session(model, loss, opt)

    report = TrainingSession.train(
        train_iter=train, 
        valid_iter=validate, 
        **defaults.training)

    evaluate.evaluation(model, data, vocab)
    
    return 0


if __name__ == '__main__':  
    print(f"Name of the script      : {argv[0]}")
    print(f"Arguments of the script : {argv[1:]}")

    try:
        exit_code = main()
    except:
        print("Exited with error")
    finally:
        print(f"{exit_code}")
