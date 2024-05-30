import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape # ts_len: the number of time steps, ts_dim: the number of variables (features)

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len) # seg_num * seg_len = ts_len
        x_embed = self.linear(x_segment) # (b d seg_num) seg_len -> (b d seg_num) d_model
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed # H which aggregates  h_(i,d) in eq.(2)
