import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer

class DecoderLayer(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, attn_ratio=1.0, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout, attn_ratio)    
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_ratio = attn_ratio
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp = self.cross_attention(
            x, cross, cross,
        )
        x = self.attn_ratio * x + self.dropout(tmp) ###
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict

class Decoder(nn.Module):
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, attn_ratio, \
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, attn_ratio, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            #print('x, cross_enc', x.shape, cross_enc.shape)
            x, layer_predict = layer(x, cross_enc)
            #print(i, x.shape, layer_predict.shape)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict # eq.(8) sigma part
            #print('final_predict', final_predict.shape)
            i += 1
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict

