# Adapted from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
import torch
import torch.nn as nn
# from .models.generator import  *
import pandas as pd
from .generator import *
import numpy as np
import random

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, encoder_q, encoder_k, dim=128, K=65536, m=0.999, T=0.07, mlp=False, generator=False):
        # def __init__(self, encoder_q, encoder_k, dim=128, K=65786, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        print('intializaitng the MoCo........................')
        self.K = K
        self.m = m
        self.T = T
        self.G = generator
        if self.G:
            print('adding generator for training')
        else:
            print('generator is disabled')
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.cal_mean = False
        if self.cal_mean:
            self.mean = torch.tensor([0])
            self.var = torch.tensor([0])
            self.total = 0

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        if self.G:
            self.generator = Generator(in_dim=256,
                                       inner_dim=256,
                                       out_dim=128)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # print(self.K, batch_size, self.K % batch_size)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        cos_similarity = 0
        if self.G:

            backbone_q = nn.Sequential(*list(self.encoder_q.children())[:-1])
            mlp_q = nn.Sequential(*list(self.encoder_q.children())[-1:])
            # print('checking the backbone', backbone_q)
            # print('checking mlq', mlp_q)
            backbone_k = nn.Sequential(*list(self.encoder_k.children())[:-1])
            mlp_k = nn.Sequential(*list(self.encoder_k.children())[-1:])
            v_q = torch.flatten(backbone_q(im_q), 1)  # queries: NxC
            q = mlp_q(v_q)
            # ratio = random.uniform(0, 1)
            ratio = random.uniform(0, 0.1)
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                v_k = torch.flatten(backbone_k(im_k), 1)  # keys: NxC
                k = mlp_k(v_k)
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                if self.cal_mean:
                    self._calculate_Mean_Val(q.detach(), k.detach())
                k_original = k
                for _ in range(1):
                    k = torch.cat((k, k_original), 0)

            q = self.generator(q, k_original, ratio)
            q = nn.functional.normalize(q, dim=1)

            cos_similarity = 0


        else:
            # compute query features
            q = self.encoder_q(im_q)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

                k = self.encoder_k(im_k)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, cos_similarity

    def cosine_similarity(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1).mean()
        return sim

    def lalign(self, x, y, alpha=2):
        return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(self, x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def _calculate_Mean_Val(self, v_q, v_k):
        # print('shape2', v_q.shape, v_k[0].shape)
        # print(torch.cat((v_q,v_k[0])).shape)
        temp_mean = torch.mean(torch.cat((v_q, v_k)), 0).to('cpu')
        # self.mean = (self.mean * self.total + temp_mean) / (self.total + 1) if self.total != 0 else self.mean
        # self.mean = self.mean.to('cpu')
        if self.total != 0:
            self.mean = (self.mean * self.total + temp_mean) / (self.total + 1)
        else:
            self.mean = temp_mean

        temp_var = torch.var(torch.cat((v_q, v_k)), 0).to('cpu')

        if self.total != 0:
            self.var = (self.var * self.total + temp_var) / (self.total + 1)
        else:
            self.var = temp_var
        self.var = self.var.to('cpu')
        self.total += 1

        if self.total % 100 == 0:
            print('max value of mean and var:', max(self.mean), max(self.var))
            print(self.mean.shape, self.var.shape)
            np.savetxt('Z_ImageNet200_mean_res18.csv', self.mean.cpu().detach().numpy(), delimiter=",")
            np.savetxt('Z_ImageNet200_var_res18.csv', self.var.cpu().detach().numpy(), delimiter=",")


# utils

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
