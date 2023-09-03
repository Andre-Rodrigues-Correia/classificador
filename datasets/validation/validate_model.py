import pickle
from nltk import word_tokenize
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
from nltk.classify.util import accuracy

from utils.utils import Utils


class Validation:

    @classmethod
    def preprocess_text(cls, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        return words

    @classmethod
    def evaluate_model_with_preset_data_test(cls):
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = Utils.get_test_dataset()
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]
        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        accuracy_score = accuracy(classifier, test_features)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)
        classification_rep = classification_report(true_labels, predicted_labels, target_names=classifier.labels())
        mcc = matthews_corrcoef(true_labels, predicted_labels)

        print("Accuracy:", accuracy_score)
        print("F1 Score:", f1)
        print("Recall:", recall)
        print("Precision:", precision)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", classification_rep)
        print("Matthews Correlation Coefficient:", mcc)

    @classmethod
    def evaluate_model_with_random_data_test(cls):
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = Utils.get_test_dataset()
        test_features = [
            ({word: True for word in cls.preprocess_text(text)}, lang) for text, lang in test_data
        ]
        true_labels = [lang for _, lang in test_features]
        predicted_labels = [classifier.classify(features) for features, _ in test_features]

        accuracy_score = accuracy(classifier, test_features)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        confusion = confusion_matrix(true_labels, predicted_labels)
        classification_rep = classification_report(true_labels, predicted_labels, target_names=classifier.labels())
        mcc = matthews_corrcoef(true_labels, predicted_labels)

        print("Accuracy:", accuracy_score)
        print("F1 Score:", f1)
        print("Recall:", recall)
        print("Precision:", precision)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", classification_rep)
        print("Matthews Correlation Coefficient:", mcc)
