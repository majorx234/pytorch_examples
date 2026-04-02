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


# add rotation to Mnist dataset
class MnistDatasetPlus(Dataset):
    def __init__(self, data):
        df_data = pandas.read_csv(data, header=None)
        self.data = [(row.iat[0], row.iloc[1:].to_numpy()) for _, row in df_data.iterrows()]

    def __len__(self):
        # len + 2x rotate left + 2xrotate right
        return 5*len(self.data)

    def __getitem__(self, index):
        # TODO add rotate -10,-5, +5, +10 degrees
        if index % 5 == 0:
            label = self.data[int(index/5)][0]
            target = pt.zeros((10))
            target[label] = 1.0
            image_values = pt.FloatTensor(self.data[int(index/5)][1]) / 255.0
        elif (index % 5 == 1):
            label = self.data[int(index/5)][0]
            target = pt.zeros((10))
            target[label] = 1.0
            image_values = pt.FloatTensor(self.data[int(index/5)][1]) / 255.0
        elif index % 5 == 2:
            label = self.data[int(index/5)][0]
            target = pt.zeros((10))
            target[label] = 1.0
            image_values = pt.FloatTensor(self.data[int(index/5)][1]) / 255.0
        elif index % 5 == 3:
            label = self.data[int(index/5)][0]
            target = pt.zeros((10))
            target[label] = 1.0
            image_values = pt.FloatTensor(self.data[int(index/5)][1]) / 255.0
        elif index % 5 == 4:
            label = self.data[int(index/5)][0]
            target = pt.zeros((10))
            target[label] = 1.0
            image_values = pt.FloatTensor(self.data[int(index/5)][1]) / 255.0
        else:
            print("error data provider")
        return image_values, target
