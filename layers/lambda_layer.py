import re
from tkinter.messagebox import NO
from turtle import pos

import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum


# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
def get_relative_position_matrix(size,):
    x = torch.repeat_interleave(torch.arange(size),size,dim=0)
    y = torch.arange(size).repeat(size)
    distance_mat = torch.stack((x,y))
    distance_mat = distance_mat[:,None, :] - distance_mat[:, :,None]
    distance_mat = torch.clamp(distance_mat, -size, size)
    distance_mat += size-1
    return distance_mat

def get_embedding(dim_k, n):
    rel_lengths = 2 * n - 1 # n = im_h = im_w the feature map size
    rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k)) # 2*n-1 2*n-1 k 
    rel_pos = get_relative_position_matrix(n)
    return rel_pos_emb[rel_pos[0], rel_pos[1]]

class LambdaLayer(nn.Module):
    def __init__(self, dim, dim_k=16, n=64, dim_out=None, heads=4, r=None):
        super().__init__()
        self.dim_out = dim_out
        self.r = r
        self.n = n
        self.k = dim_k
        assert(dim_out % heads) == 0
        self.heads = heads
        dim_v = dim_out // heads

# (Page 5) BN after Q and V are helpful
        self.get_q = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim_k*heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_k*heads),
        )
        self.get_v = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim_v, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_v),
        )
        self.get_k = nn.Conv2d(in_channels=dim, out_channels=dim_k, kernel_size=1, bias=False)

        if r is not None:
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(1, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else:
            # rel_lengths = 2 * n - 1 # n = im_h = im_w the feature map size
            # self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k)) # n m k 
            self.rel_pos = get_relative_position_matrix(n)



    def forward(self,x):
        (b, c, im_h, im_w) = x.shape
        Q = self.get_q(x)
        K = self.get_k(x)
        V = self.get_v(x)

        # m = n = im_h * im_w
        Q = rearrange(Q, 'b (h k) im_h im_w -> b h (im_h im_w) k', h=self.heads) # b h n(m) k 
        K = rearrange(K, 'b k im_h im_w -> b (im_h im_w) k') # b m k 
        V = rearrange(V, 'b v im_h im_w -> b (im_h im_w) v') # b m v

        σ_K = K.softmax(dim=-1)
        λc = einsum('b m k, b m v -> b k v', σ_K, V)
        content_output = einsum('b h n k, b k v -> b n h v', Q, λc)

        if self.r is not None:
            V = rearrange(V, 'b m v -> b m v p', p = 1)
            embeddings = self.pos_conv(V)
            position_output = einsum('b h k n, b k v n -> b n h v', Q, embeddings.flatten(3))
        else:
            embeddings = get_embedding(self.k, im_h) # n m k
            λp = einsum('n m k, b m v -> b n k v', embeddings, V)
            position_output = einsum('b h n k, b n k v -> b n h v',Q, λp)

        output = rearrange(content_output + position_output, 'b (im_h im_w) h v -> b im_h im_w (h v)', im_h=im_h, im_w=im_w)
        
        return output
