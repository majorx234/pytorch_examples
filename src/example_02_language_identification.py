import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

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


def text2trigram_tensor(text, trigram_sub_catalog_dict):
    trigrams = triplet_tokenizer(text, clear_spaces=True, ascii_only=True, latanize=True)
    trigram_tensor = torch.Tensor([0]*len(trigram_sub_catalog_dict))
    for trigram in trigrams:
        if trigram in trigram_sub_catalog_dict:
            trigram_tensor[trigram_sub_catalog_dict[trigram]] = 1.0
    return trigram_tensor


def lang2language_tensor(lang, lang_filter_dict):
    language_tensor = torch.Tensor([0]*len(lang_filter_dict))
    if lang in lang_filter_dict:
        idx = lang_filter_dict[lang]
        language_tensor[idx] = 1.0
    return language_tensor


class LanguageIdentifier():
    def __init__(self, train_data_csv, valid_data_csv, test_data_csv, lang_filter):
        lang_filter_dict = {}
        for idx, item in enumerate(lang_filter):
            lang_filter_dict[item] = idx
        self.lang_filter_dict = lang_filter_dict
        self.train_data = LanguageDataset(csv_file_to_rows(train_data_csv, lang_filter))
        self.valid_data = LanguageDataset(csv_file_to_rows(valid_data_csv, lang_filter))
        self.test_data = LanguageDataset(csv_file_to_rows(test_data_csv, lang_filter))
        self.prepared = False

    def prepare(self):
        # create trigram catalog
        nsamples_trigram_catalog = 100
        lang_ident_range = range(len(self.train_data))
        lang_ident_random_idexes = random.sample(lang_ident_range, nsamples_trigram_catalog)
        trigram_catalog_dict = {}
        trigram_catalog_list = []
        ntrigram_index = 0
        for idx in lang_ident_random_idexes:
            trigrams = triplet_tokenizer(self.train_data[idx][1], clear_spaces=True, ascii_only=True, latanize=True)
            for trigram in trigrams:
                if trigram not in trigram_catalog_dict:
                    trigram_catalog_dict[trigram] = ntrigram_index
                    trigram_catalog_list.append(trigram)
                    ntrigram_index += 1

        # from all trigrams take subset of 768
        ntrigrams = len(trigram_catalog_dict)
        trigram_subset_size = 768
        trigram_subset_index = random.sample(range(0, ntrigrams), trigram_subset_size)

        self.trigram_sub_catalog_dict = {}
        self.trigram_sub_catalog_list = []
        sub_idx = 0
        for idx in trigram_subset_index:
            self.trigram_sub_catalog_dict[trigram_catalog_list[idx]] = sub_idx
            self.trigram_sub_catalog_list.append(trigram_catalog_list[idx])
            sub_idx += 1

        # create KNN
        self.knn = FeedForward01(trigram_subset_size, 50, len(self.lang_filter_dict))
        self.criterion = nn.CrossEntropyLoss()
        learning_rate = 0.0005
        self.optimizer = optim.SGD(self.knn.parameters(), lr=learning_rate)

    def train_step(self, text_triplet_tensor, language_tensor):
        # set model in training mode
        self.knn.train()
        self.optimizer.zero_grad()
        output = self.knn.forward(text_triplet_tensor)
        loss = self.criterion(output, language_tensor)
        # calculate backward propagation
        loss.backward()
        # optimize parameters of KNN
        self.optimizer.step()

        return output, loss.item()

    def train(self, epoch_size=10):
        for epoch in tqdm(range(epoch_size), unit="epoch"):
            samples = list(range(len(self.train_data)))
            random.shuffle(samples)
            for idx in tqdm(samples, unit="samples", desc="training"):
                language, text = self.train_data
                language_tensor = lang2language_tensor(language, self.lang_filter_dict)
                trigram_tensor = text2trigram_tensor(text, self.trigram_sub_catalog_dict)
                output, loss = self.train_step(trigram_tensor, language_tensor)



def main():
    lang_filter = ["nl", "es", "it", "tr", "fr", "pl", "pt", "en", "de", "sw"]
    knn_lang_ident = LanguageIdentifier("../data/train.csv",
                                        "../data/valid.csv",
                                        "../data/test.csv",
                                        lang_filter)
    print(knn_lang_ident)


if __name__ == "__main__":
    main()
