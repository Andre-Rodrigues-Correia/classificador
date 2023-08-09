from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from utils.utils import Utils


class Train:
    INDEX_SENTENCE = 1
    INDEX_LANGUAGE = 0

    @classmethod
    def classify_using_trained_dataset(cls, test_text):
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        words = cls.preprocess_text(test_text)
        features = {word: True for word in words}
        predicted_lang = classifier.classify(features)
        print("Texto:", test_text)
        print("Língua Predita:", predicted_lang)

    @classmethod
    def get_accuracy(cls):
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = Utils.get_test_dataset()
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]
        accuracy_score = accuracy(classifier, test_features)
        print("Acurácia do Modelo:", accuracy_score)

    @classmethod
    def test_matriz(cls):
        load_training = open('datasets/model.pickle', 'rb')
        test_data = Utils.get_test_dataset()
        classifier = pickle.load(load_training)
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]
        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        # Calcular a matriz de confusão
        confusion = confusion_matrix(true_labels, predicted_labels)

        accuracy_score = accuracy(classifier, test_features)

        # Visualizar a matriz de confusão usando Seaborn e Matplotlib
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

        training_features = []
        for text, lang in train_set:
            words = cls.preprocess_text(text)
            features = {word: True for word in words}
            training_features.append((features, lang))

        # Treinar o classificador Naive Bayes
        classifier = NaiveBayesClassifier.train(training_features)

        # Exemplo de texto para classificar
        test_text = "Esse é um teste para verificar a classificação."

        # Pré-processamento do texto de teste
        test_words = cls.preprocess_text(test_text)
        test_features = {word: True for word in test_words}

        # Classificar o exemplo de texto
        predicted_lang = classifier.classify(test_features)
        print("Texto:", test_text)
        print("Língua Predita:", predicted_lang)

        # Avaliar a precisão do classificador
        accuracy_score = accuracy(classifier, training_features)
        print("Precisão do Classificador:", accuracy_score)

        save_training = open('datasets/model.pickle', 'wb')
        pickle.dump(classifier, save_training)  # SAVE TRAINED CLASSIFIER
        save_training.close()