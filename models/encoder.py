import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        # ts_d: the number of variables
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, attn_ratio, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm) # eq.(6)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, \
                                                        d_ff, dropout, attn_ratio))
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x = layer(x) # eq.(5)       
        
        return x

class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout, attn_ratio,
                in_seg_num = 10, factor=10):
        # e_blocks: the number of encoder layers
        # in_seg_num = pad_out_len (the number of time steps + padding) // seg_len
        # win_size: the number of segments for segment merging
        # factor: the number of routers  in Cross-Dimension stage
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        # encoder layer 1 does not need segment merging
        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout, attn_ratio, \
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout, attn_ratio, \
                                            ceil(in_seg_num/win_size**i), factor))

    def forward(self, x):
        encode_x = [] # Add outputs of each encoder layer
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
