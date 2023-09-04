import pickle

from nltk import word_tokenize
from nltk.classify.util import accuracy
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import Utils


class Tests:
    AMOUNT_VALIDATIONS = 10

    @classmethod
    def get_accuracies(cls):
        for i in range(10):
            cls.test_accuracy_with_random_texts_test_dataset()
            cls.test_accuracy_with_random_texts_test_dataset_with_cities()
            print('\n-------\n')

    @classmethod
    def get_metrics(cls):
        for i in range(5):
            cls.get_metrics_with_default_dataset(f"default_dataset_validate_{i}")
            cls.get_metrics_with_cities_feature_dataset(f"cities_feature_dataset_validate_{i}")
            print('\n-------\n')

    @classmethod
    def test_accuracy_with_random_texts_test_dataset(cls):
        total_accuracy_score = 0.0
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        for i in range(cls.AMOUNT_VALIDATIONS):
            test_data = Utils.get_random_test_dataset()
            test_features = [
                ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
            ]
            total_accuracy_score += accuracy(classifier, test_features)

        print("Average Precision default dataset: ", (total_accuracy_score / cls.AMOUNT_VALIDATIONS))

    @classmethod
    def test_accuracy_with_random_texts_test_dataset_with_cities(cls):
        total_accuracy_score = 0.0
        with open("datasets/cities/brazil_cities.txt", encoding="utf8") as cities_brasil_file:
            cities_brasil = set(city.strip() for city in cities_brasil_file)

        with open("datasets/cities/portugal_cities.txt", encoding="utf8") as cities_portugal_file:
            cities_portugal = set(city.strip() for city in cities_portugal_file)
        load_training = open('datasets/model_with_cities.pickle', 'rb')

        classifier = pickle.load(load_training)

        for i in range(cls.AMOUNT_VALIDATIONS):
            test_data = Utils.get_random_test_dataset()
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

            total_accuracy_score += accuracy(classifier, test_features)

        print("Average Precision recognition_cities_dataset: ", (total_accuracy_score / cls.AMOUNT_VALIDATIONS))


    @classmethod
    def get_metrics_with_default_dataset(cls, output_filename):
        total_accuracy_score = 0.0
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = Utils.get_random_test_dataset()
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]
        accuracy_score = accuracy(classifier, test_features)

        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)
        mcc = matthews_corrcoef(true_labels, predicted_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classifier.labels(), yticklabels=classifier.labels())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(
            f'Accuracy: {accuracy_score}\n'
            f'F1 Score: {f1}\n'
            f'Precision: {precision}\n'
            f'Recall: {recall}\n'
            f'Matthews Correlation Coefficient: {mcc}\n\n'
            f'Confusion Matrix'
        )
        plt.savefig('datasets/validation/confusion_matriz/{}'.format(output_filename), bbox_inches='tight')
        plt.close()
        print("Accuracy default dataset: ", accuracy_score)

    @classmethod
    def get_metrics_with_cities_feature_dataset(cls, output_filename):
        total_accuracy_score = 0.0
        with open("datasets/cities/brazil_cities.txt", encoding="utf8") as cities_brasil_file:
            cities_brasil = set(city.strip() for city in cities_brasil_file)

        with open("datasets/cities/portugal_cities.txt", encoding="utf8") as cities_portugal_file:
            cities_portugal = set(city.strip() for city in cities_portugal_file)
        load_training = open('datasets/model_with_cities.pickle', 'rb')

        classifier = pickle.load(load_training)

        test_data = Utils.get_random_test_dataset()
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

        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)
        mcc = matthews_corrcoef(true_labels, predicted_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                        xticklabels=classifier.labels(), yticklabels=classifier.labels())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(
            f'Accuracy: {accuracy_score}\n'
            f'F1 Score: {f1}\n'
            f'Precision: {precision}\n'
            f'Recall: {recall}\n'
            f'Matthews Correlation Coefficient: {mcc}\n\n'
            f'Confusion Matrix'
        )
        plt.savefig('datasets/validation/confusion_matriz/{}'.format(output_filename), bbox_inches='tight')
        plt.close()

        print("Accuracy cities feature dataset: ", accuracy_score)

    @classmethod
    def preprocess_text(cls, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        return words
