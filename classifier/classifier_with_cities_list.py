import pickle
from train.train import Train


class ClassifierWithCitiesList:

    @classmethod
    def classifier_text(cls, text):
        load_training = open('datasets/model_with_cities.pickle', 'rb')
        classifier = pickle.load(load_training)
        words = Train.preprocess_text(text)
        features = {word: True for word in words}
        predicted_lang = classifier.classify(features)
        return predicted_lang
