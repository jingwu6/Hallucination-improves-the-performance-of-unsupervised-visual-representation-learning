# Adapted from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
import torch.nn as nn
from .generator import *
import random

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, encoder, dim=2048, pred_dim=512, generator=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.G = generator

        if self.G:
            print('adding generator for training')
            self.generator = Generator(in_dim=1024,
                                       inner_dim=1024,
                                       out_dim=512)
        else:
            print('generator is disabled')
        self.encoder = encoder

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC


        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        if self.G:
            for _ in range(1):
                ratio = random.uniform(0, 0.1)
                p1_origin = p1
                p1 = self.generator(p1_origin, p2, ratio)
                ratio = random.uniform(0, 0.1)
                p2 = self.generator(p2, p1_origin, ratio)


                z1 = torch.cat((z1, z1), 0)
                z2 = torch.cat((z2, z2), 0)
                assert p1.shape == z2.shape and p2.shape == z1.shape, 'ensure hallucinated samples shape'


        return p1, p2, z1.detach(), z2.detach()
