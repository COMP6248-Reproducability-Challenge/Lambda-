import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum

# def lambda layer(queries, keys, embeddings, values):
# """Multi−query lambda layer."""
# # b: batch, n: input length, m: context length,
# # k: query/key depth, v: value depth,
# # h: number of heads, d: output dimension.
# content lambda = einsum(softmax(keys), values, ’bmk,bmv−>bkv’)
# position lambdas = einsum(embeddings, values, ’nmk,bmv−>bnkv’)
# content output = einsum(queries, content lambda, ’bhnk,bkv−>bnhv’)
# position output = einsum(queries, position lambdas, ’bhnk,bnkv−>bnhv’)
# output = reshape(content output + position output, [b, n, d])
# return output

# ref: https://www.programcreek.com/python/?CodeExample=generate+relative+positions+matrix
def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    # 以下代码： 主对角线为0
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    # 以下代码，主对角线为维度值
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat 

class LambdaLayer(nn.Module):
    def __init__(self, dim, dim_k, n, dim_out, heads):
        super(LambdaLayer, self).__init__()

        assert dim_out.size == dim.size
        assert(dim_out % heads) == 0
        self.heads = heads
        dim_v = dim_out // heads

# (Page 5) BN after Q and V are helpful
        self.get_q = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim_k*heads, stride=1, bias=False),
            nn.BatchNorm2d(dim_k*heads),
        )
        self.get_v = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim_v, stride=1, bias=False),
            nn.BatchNorm2d(dim_v),
        )
        self.get_k = nn.Conv2d(in_channels=dim, out_channels=dim_k*heads, stride=1, bias=False)

        ## TODO： 
        self.embedding = ()
        self.relative_position = generate_relative_positions_matrix(dim)

    def forward(self,x):
        (b, c, im_h, im_w) = x.shape
        Q = self.get_q(x)
        K = self.get_k(x)
        V = self.get_v(x)
        
        Q = rearrange(Q, 'b (h k) im_h im_w -> b h k (im_h im_w)', h=self.heads)
        K = rearrange(K, 'b k im_h im_w -> b k (im_h im_w)')
        V = rearrange(V, 'b v im_h im_w -> b v (im_h im_w)')

        σ_K = nn.Softmax(K, dim=-1)
        λc = einsum('b k m, b v m -> b k v', σ_K, V)
        λp = einsum('b ') ## TODO: 添加Embedding

        λn = λc + λp
