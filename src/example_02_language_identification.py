import random

from .example_01_simple_feed_forward import FeedForward01
from .tokenizer import triplet_tokenizer
from .language_dataset import LanguageDataset


def csv_file_to_rows(file_name, lang_filter):
    first = True
    with open(file_name, 'r') as file:
        rows = []
        for line in file:
            if first:
                first = False
                continue
            splitline = line.split(",", 1)
            if len(splitline) == 2:
                if (splitline[0] in lang_filter):
                    rows.append((splitline[0], splitline[1]))
            else:
                print(splitline)
        return rows


class LanguageIdentifier():
    def __init__(self, train_data_csv, valid_data_csv, test_data_csv, lang_filter):
        self.train_data = LanguageDataset(csv_file_to_rows(train_data_csv, lang_filter))
        self.valid_data = LanguageDataset(csv_file_to_rows(valid_data_csv, lang_filter))
        self.test_data = LanguageDataset(csv_file_to_rows(test_data_csv, lang_filter))


def main():
    nsamples_trigram_catalog = 100
    lang_filter = ["nl", "es", "it", "tr", "fr", "pl", "pt", "en", "de", "sw"]
    knn_lang_ident = LanguageIdentifier("../data/train.csv",
                                        "../data/valid.csv",
                                        "../data/test.csv",
                                        lang_filter)
    print(knn_lang_ident)
    # create trigram catalog
    lang_ident_range = range(len(knn_lang_ident.train_data))
    lang_ident_random_idexes = random.sample(lang_ident_range, nsamples_trigram_catalog)
    trigram_catalog_dict = {}
    trigram_catalog_list = []
    ntrigram_index = 0
    for idx in lang_ident_random_idexes:
        trigrams = triplet_tokenizer(knn_lang_ident.train_data[idx][1], clear_spaces=True, ascii_only=True, latanize=True)
        for trigram in trigrams:
            if trigram not in trigram_catalog_dict:
                trigram_catalog_dict[trigram] = ntrigram_index
                trigram_catalog_list.append(trigram)
                ntrigram_index += 1

    # from all trigrams take subset of 768
    ntrigrams = len(trigram_catalog_dict)
    trigram_subset_index = random.sample(range(0, ntrigrams), 768)

    trigram_sub_catalog_dict = {}
    trigram_sub_catalog_list = []
    sub_idx = 0
    for idx in trigram_subset_index:
        trigram_sub_catalog_dict[trigram_catalog_list[idx]] = sub_idx
        trigram_sub_catalog_list.append(trigram_catalog_list[idx])
        sub_idx += 1


if __name__ == "__main__":
    main()
