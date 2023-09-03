import pickle

from nltk import accuracy, word_tokenize

from train.train import Train
from utils.utils import Utils


class Classifier:

    @classmethod
    def classifier_text(cls, text):
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        words = Train.preprocess_text(text)
        features = {word: True for word in words}
        predicted_lang = classifier.classify(features)
        probabilities = classifier.prob_classify(features)
        probability_correct = probabilities.prob(predicted_lang)
        return predicted_lang, probability_correct
