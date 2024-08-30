
import torch
import torch.nn as nn

class MLP(nn.Module):
    """ A simple MLP. """

    def __init__(self, d_in, hidden_sizes, activation=nn.GELU()):
        super().__init__()
        self.d_in = d_in
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.layers = nn.ModuleList()
        for i, h in enumerate(hidden_sizes):
            h_in = d_in if i == 0 else hidden_sizes[i - 1]
            h_out = h
            self.layers.append(nn.Linear(h_in, h_out))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        # No activation on final layer
        x = self.layers[-1](x)
        return x
