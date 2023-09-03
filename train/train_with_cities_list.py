import io

from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from utils.utils import Utils


class TrainWithCitiesList:

    INDEX_SENTENCE = 1
    INDEX_LANGUAGE = 0

    @classmethod
    def get_accuracy(cls):
        with open("datasets/cities/brazil_cities.txt", encoding="utf8") as cities_brasil_file:
            cities_brasil = set(city.strip() for city in cities_brasil_file)

        with open("datasets/cities/portugal_cities.txt", encoding="utf8") as cities_portugal_file:
            cities_portugal = set(city.strip() for city in cities_portugal_file)
        load_training = open('datasets/model_with_cities.pickle', 'rb')

        classifier = pickle.load(load_training)
        test_data = Utils.get_test_dataset()
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]

        for text, lang in test_data:
            words = cls.preprocess_text(text)
            features = {word: True for word in words}

            # Check for city names in the text and adjust the feature set accordingly
            for city in cities_brasil:
                if city in text:
                    features[f"brasil_city_{city}"] = 2  # Increase weight for city words

            for city in cities_portugal:
                if city in text:
                    features[f"portugal_city_{city}"] = 2  # Increase weight for city words

            test_features.append((features, lang))
        accuracy_score = accuracy(classifier, test_features)
        return accuracy_score

    @classmethod
    def create_confusion_matriz(cls):
        load_training = open('datasets/model_with_cities.pickle', 'rb')
        test_data = Utils.get_test_dataset()
        classifier = pickle.load(load_training)
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]
        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        confusion = confusion_matrix(true_labels, predicted_labels)

        accuracy_score = accuracy(classifier, test_features)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classifier.labels(), yticklabels=classifier.labels())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy_score:.2f}')
        plt.show()

    @classmethod
    def preprocess_text(cls, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        return words

    @classmethod
    def train_model(cls):
        train_set = []

        with open("datasets/formated_mid_dataset.txt", encoding="utf8") as read_file:
            for sentence in read_file:
                splited_sentences = sentence.split('\t')
                train_set.append((splited_sentences[cls.INDEX_SENTENCE], splited_sentences[cls.INDEX_LANGUAGE]))

                # Load city names from files
        with open("datasets/cities/brazil_cities.txt", encoding="utf8") as cities_brasil_file:
            cities_brasil = set(city.strip() for city in cities_brasil_file)

        with open("datasets/cities/portugal_cities.txt", encoding="utf8") as cities_portugal_file:
            cities_portugal = set(city.strip() for city in cities_portugal_file)

        training_features = []
        for text, lang in train_set:
            words = cls.preprocess_text(text)
            features = {word: True for word in words}

            # Check for city names in the text and adjust the feature set accordingly
            for city in cities_brasil:
                if city in text:
                    features[f"brasil_city_{city}"] = 2  # Increase weight for city words

            for city in cities_portugal:
                if city in text:
                    features[f"portugal_city_{city}"] = 2  # Increase weight for city words

            training_features.append((features, lang))

        classifier = NaiveBayesClassifier.train(training_features)

        test_text = "Esse é um teste para verificar a classificação."

        test_words = cls.preprocess_text(test_text)
        test_features = {word: True for word in test_words}

        predicted_lang = classifier.classify(test_features)
        print("Texto:", test_text)
        print("Língua Predita:", predicted_lang)

        accuracy_score = accuracy(classifier, training_features)
        print("Precisão do Classificador:", accuracy_score)

        save_training = open('datasets/model_with_cities.pickle', 'wb')
        pickle.dump(classifier, save_training)  # SAVE TRAINED CLASSIFIER
        save_training.close()
