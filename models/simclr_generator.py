from .generator import *
import torch
import torch.distributed
from torch import nn, optim
import random


class simclr_generator(nn.Module):

    def __init__(self, backbone, generator=Generator):

        super(simclr_generator, self).__init__()

        self.backbone = backbone

        self.generator = generator(in_dim=256, inner_dim=256, out_dim=128)

    def forward(self, x):
        batch_size = x.shape[0] // 2
        q = self.backbone(x[0:batch_size])
        k = self.backbone(x[batch_size:])
        ratio = random.uniform(0, 0.1)
        q = self.generator(q, k, ratio)
        k_original = k
        for _ in range(1):
            k = torch.cat((k, k_original), 0)

        return torch.cat((q, k), 0)