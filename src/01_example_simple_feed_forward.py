import torch.nn as nn


class FeedForward01(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward01, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.hidden = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_1 = self.input_layer(x)
        y_1_activation = self.sigmoid(y_1)
        out = self.hidden(y_1_activation)

        return out
