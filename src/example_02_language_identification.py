from .example_01_simple_feed_forward import FeedForward01
from .tokenizer import triplet_tokenizer
from .language_dataset import LanguageDataset


def csv_file_to_rows(file_name, filter):
    first = True
    with open(file_name, 'r') as file:
        rows = []
        for line in file:
            if first:
                first = False
                continue
            splitline = line.split(",", 1)
            if len(splitline) == 2:
                if (splitline[0] in filter):
                    rows.append((splitline[0], splitline[1]))
            else:
                print(splitline)
        return rows


class LanguageIdentifier():
    def __init__(self, train_data_csv, valid_data_csv, test_data_csv, filter, ascii_only=False):
        self.train_data = LanguageDataset(csv_file_to_rows(train_data_csv))
        self.valid_data = LanguageDataset(csv_file_to_rows(valid_data_csv))
        self.test_data = LanguageDataset(csv_file_to_rows(test_data_csv))


def main():
    filter = ["nl", "es", "it", "tr", "fr", "pl", "pt", "en", "vi", "de", "sw"]
    knn_lang_ident = LanguageIdentifier("../data/train.csv",
                                        "../data/valid.csv",
                                        "../data/test.csv",
                                        filter)
    print(knn_lang_ident)


if __name__ == "__main__":
    main()
