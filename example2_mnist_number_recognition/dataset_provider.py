import pandas
from torch.utils.data import Dataset
import torch as pt


class MnistDataset(Dataset):
    def __init__(self, data):
        df_data = pandas.read_csv(data, header=None)
        self.data = [(row.iat[0], row.iloc[1:].to_numpy()) for _, row in df_data.iterrows()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data[index][0]
        target = pt.zeros((10))
        target[label] = 1.0
        image_values = pt.FloatTensor(self.data[index][1]) / 255.0
        return image_values, target
