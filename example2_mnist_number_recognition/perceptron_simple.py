import torch as pt
import torch.nn as nn
from torch.utils.data import DataLoader
from mnist_dataset import MnistDataset


class PerceptronSimple(nn.Module):
    def __init__(self, lr=0.01):
        super(PerceptronSimple, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        self.loss_function = nn.MSELoss()
        self.optimizer = pt.optim.SGD(self.parameters(), lr=lr)

    def forward(self, input):
        return self.model(input)


class PerceptronSimpleTrainer:
    def __init__(self, train_data, test_data, learn_rate):
        self.train_dataset = MnistDataset(train_data)
        self.test_dataset = MnistDataset(test_data)
        self.classifier = PerceptronSimple(lr=learn_rate)
        self.loss_fn = nn.BCELoss()

    def prepare(self):
        # prepare datset
        print("TODO prepare")

    def _train_step(self):
        # just one train step of training
        print("TODO: train_step()")

    def train(self, n_epochs=40, batch_size=60):
        # just one train step of training
        print("train")
        train_dataloader = DataLoader(self.train_dataset, batch_size=60, shuffle=True)
        for epoch in range(n_epochs):
            # loop over dataset, in batchsize steps
            for images, targets in train_dataloader:
                # outputs save the computation path
                outputs = self.classifier(images)
                # compute the loss
                loss = self.loss_fn(outputs, targets)
                # backward pass
                loss.backward()
                # compute gradients for all parameters

                # apply the gradients to update model weights
                self.classifier.optimizer.step()
                # clear gradients for the next iteration
                self.classifier.optimizer.zero_grad()
