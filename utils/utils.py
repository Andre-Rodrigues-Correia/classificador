import re


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
    def format_file_from_train(cls, language, directory):
        with open("datasets/formated_mid_dataset.txt".format(language), "a+", encoding="utf-8") as write_file:
            with open(directory, encoding="utf8") as read_file:
                for sentence in read_file:
                    splited_sentences = sentence.split('\t')
                    normalized_text = cls.normalize_text(splited_sentences[cls.INDEX_SENTENCE])
                    write_file.write('{}\t{}'.format(language, normalized_text))
