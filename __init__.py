from sys import argv
from torch import cuda
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed

from config import defaults, genconfig
from src import preprocess, training, evaluate
from src.model import lstm, transformer


def main(model = "transformer", dataset = "toy-ende"):    
    init_logger()
    is_cuda = cuda.is_available()
    set_random_seed(1111, is_cuda)

    data = preprocess.setup_dataset(dataset)
    vocab = preprocess.setup_vocab(data)

    model, loss, opt = None, None, None
    if model == "transformer":
        model, loss, opt = transformer.SimpleTransformer(vocab)
    elif model == "lstm":
        model, loss, opt = lstm.BaseLSTMModel(vocab)

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
        params = genconfig.gen_defaults_config(argv[1:])
        exit_code = main(**params)
    except:
        print("[Exited with error]")
    else:
        print("[Exited gracefully]")
