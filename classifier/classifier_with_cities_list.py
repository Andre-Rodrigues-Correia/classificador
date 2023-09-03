import pickle

from nltk import word_tokenize

from train.train_with_cities_list import TrainWithCitiesList


class ClassifierWithCitiesList:

    @classmethod
    def classifier_text(cls, text):
        load_training = open('datasets/model_with_cities.pickle', 'rb')
        classifier = pickle.load(load_training)
        words = TrainWithCitiesList.preprocess_text(text)

        with open("datasets/cities/brazil_cities.txt", encoding="utf8") as cities_brasil_file:
            cities_brasil = set(city.strip() for city in cities_brasil_file)

        with open("datasets/cities/portugal_cities.txt", encoding="utf8") as cities_portugal_file:
            cities_portugal = set(city.strip() for city in cities_portugal_file)

        features = {word: True for word in words}

        for city in cities_brasil:
            if city in text:
                features[f"brasil_city_{city}"] = 2  # Increase weight for city words

        for city in cities_portugal:
            if city in text:
                features[f"portugal_city_{city}"] = 2  # Increase weight for city words

        predicted_lang = classifier.classify(features)
        probabilities = classifier.prob_classify(features)
        probability_correct = probabilities.prob(predicted_lang)
        return predicted_lang, probability_correct

    @classmethod
    def preprocess_text(cls, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        return words

    @classmethod
    def classifier_text_2(cls, text):
        load_training = open('datasets/model_with_cities.pickle', 'rb')
        classifier = pickle.load(load_training)
        words = TrainWithCitiesList.preprocess_text(text)

        with open("datasets/cities/brazil_cities.txt", encoding="utf8") as cities_brasil_file:
            cities_brasil = set(city.strip() for city in cities_brasil_file)

        with open("datasets/cities/portugal_cities.txt", encoding="utf8") as cities_portugal_file:
            cities_portugal = set(city.strip() for city in cities_portugal_file)

        features = {word: True for word in words}

        for city in cities_brasil:
            if city in text:
                features[f"brasil_city_{city}"] = 2  # Increase weight for city words

        for city in cities_portugal:
            if city in text:
                features[f"portugal_city_{city}"] = 2  # Increase weight for city words

        predicted_lang = classifier.classify(features)
        probabilities = classifier.prob_classify(features)
        probability_correct = probabilities.prob(predicted_lang)
        return predicted_lang, probability_correct
