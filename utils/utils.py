import pickle
import re
import random
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef
from nltk.classify.util import accuracy



class Utils:
    INDEX_SENTENCE = 1
    INDEX_LANGUAGE = 0

    @classmethod
    def normalize_text(cls, text):
        normalized_text = re.sub(r'[^\w\s]', '', text)
        print(normalized_text)
        return normalized_text

    @classmethod
    def get_test_dataset(cls):
        test_dataset = []
        with open("datasets/validate_dataset.txt", encoding="utf8") as read_file:
            for sentence in read_file:
                splited_sentences = sentence.split('\t')
                test_dataset.append((splited_sentences[cls.INDEX_SENTENCE], splited_sentences[cls.INDEX_LANGUAGE]))
        return test_dataset

    @classmethod
    def get_random_test_dataset(cls):
        num_lines_to_sample = 2000
        test_dataset = []

        with open("datasets/validate_dataset.txt", encoding="utf8") as read_file:
            lines = read_file.readlines()
            num_lines = len(lines)

            if num_lines_to_sample >= num_lines:
                return [(line.split('\t')[1], line.split('\t')[0]) for line in lines]

            sampled_indices = random.sample(range(num_lines), num_lines_to_sample)
            for index in sampled_indices:
                line = lines[index]
                splited_sentences = line.split('\t')
                test_dataset.append((splited_sentences[1], splited_sentences[0]))

        return test_dataset

    @classmethod
    def format_file_from_train(cls, language, directory):
        with open("datasets/formated_mid_dataset.txt".format(language), "a+", encoding="utf-8") as write_file:
            with open(directory, encoding="utf8") as read_file:
                for sentence in read_file:
                    splited_sentences = sentence.split('\t')
                    normalized_text = cls.normalize_text(splited_sentences[cls.INDEX_SENTENCE])
                    write_file.write('{}\t{}'.format(language, normalized_text))

    @classmethod
    def get_portugal_genetics(cls):
        url = "https://pt.wikipedia.org/wiki/Lista_de_gent%C3%ADlicos_de_Portugal"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        gentilicos = []

        # Encontre a tabela com os gentílicos
        table = soup.find("table", {"class": "wikitable"})

        # Percorra as linhas da tabela
        for row in table.find_all("tr")[1:]:
            columns = row.find_all("td")
            if len(columns) >= 2:
                pais = columns[0].get_text(strip=True)
                gentilico = columns[1].get_text(strip=True)
                gentilicos.append((pais, gentilico))

        # Exibir os gentílicos obtidos
        for pais, gentilico in gentilicos:
            print(f"{pais}: {gentilico}")

    @classmethod
    def get_brazil_genetics(cls):
        # https://github.com/livyreal/BR_Gentilics/blob/master/Cities%20of%20Paran%C3%A1%20Pernambuco%20Piaui.csv
        return 0

    @classmethod
    def preprocess_text(cls, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        return words

    @classmethod
    def evaluate_model(cls):
        load_training = open('datasets/model.pickle', 'rb')
        classifier = pickle.load(load_training)
        test_data = cls.get_test_dataset()
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
        #roc_auc = roc_auc_score(true_labels, predicted_labels)
        mcc = matthews_corrcoef(true_labels, predicted_labels)

        print("Accuracy:", accuracy_score)
        print("F1 Score:", f1)
        print("Recall:", recall)
        print("Precision:", precision)
        print("Confusion Matrix:\n", confusion)
        print("Classification Report:\n", classification_rep)
        print("Matthews Correlation Coefficient:", mcc)
