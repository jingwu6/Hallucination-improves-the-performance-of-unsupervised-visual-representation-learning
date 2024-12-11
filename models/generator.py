import sys
from argparse import ArgumentParser
from itertools import chain
import torch
import torch.distributed
from torch import nn, optim



class Generator(nn.Module):

    def __init__(self, in_dim=1024, inner_dim=1024, out_dim=512):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(in_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.fc3 = nn.Linear(inner_dim, out_dim)
        self.in_dim = in_dim


        self.MLP = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

    def forward(self, x, k, ratio):
        for i in range(1):
            generated = (1 + ratio) * x - ratio * k
            x1 = torch.cat((x, generated), 1)
            generated = self.MLP(x1)
            out = torch.cat((x, generated), 0)

        return out


if __name__ == '__main__':
    mean = abs(torch.randn(512))
    generator = Generator(mean, mean, batch_size=2)
    test = torch.randn(2, 512)

    print(generator(test).shape)
