from train.train import Train
from train.prepare_training import prepare_models
import _pickle

# é a primeira vez executando o códifo ? se sim defina a variável 'IS_FIRST' como true
IS_FIRST = False

if __name__ == '__main__':
    if(IS_FIRST):
        prepare_models()
        Train.train_model()

    #Train.classify_using_trained_dataset("Rio de janeiro")
    #Train.get_accuracy()
    Train.test_matriz()


