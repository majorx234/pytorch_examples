from .example_01_simple_feed_forward import FeedForward01
from .tokenizer import triplet_tokenizer
from .language_dataset import LanguageDataset


def csv_file_to_rows(file_name):
    first = false
    with open(file_name, 'r') as file:
        rows = []
        for line in file:
            if first:
                first = False
                continue
            splitline = line.split(",", 1)
            rows.append((splitline[0], splitline[1]))
        return rows


class LanguageIdentifier():
    def __init__(self, train_data_csv, valid_data_csv, test_data_csv):
        self.train_data = csv_file_to_rows(train_data_csv)
        self.valid_data = csv_file_to_rows(valid_data_csv)
        self.test_data = csv_file_to_rows(test_data_csv)

