import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum, nn

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

def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat 

class LambdaLayer(nn.Module):
    def __init__(self, dim, dim_k, n, dim_out, heads) -> None:
        super(LambdaLayer, self).__init__()
        assert dim_out.size == dim.size
        assert(dim_out % heads) == 0

        self.heads = heads
        dim_v = dim_out // heads

        self.get_q = nn.Conv2d(in_channels=dim, out_channels=dim_k*heads, stride=1, bias=False)
        self.get_k = nn.Conv2d(in_channels=dim, out_channels=dim_k*heads, stride=1, bias=False)
        self.get_v = nn.Conv2d(in_channels=dim, out_channels=dim_v, stride=1, bias=False)
        