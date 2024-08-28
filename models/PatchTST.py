__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from layers.PatchTST_backbone import PatchTST_backbone,PatchTST_backbone_decom
from layers.PatchTST_layers import series_decomp
from layers.SelfAttention_Family import AttentionLayer1, FullAttention, ProbAttention
def split_and_pad_sequence(x, p, stride):
    b, c, t = x.shape
    n = (t - p) // stride + 1  # 计算切分后的段数
    # 创建结果张量
    result = torch.zeros((b, c, n, p)).to(x.device)
    for i in range(n):
        start = i * stride
        end = start + p
        # 切分序列
        segment = x[:, :, start:end]
        # 处理不够长的部分，用t维度的平均值进行填充
        if end > t:
            pad_length = end - t
            average_value = torch.mean(x[:, :, start:t], dim=2, keepdim=True)
            pad_values = average_value.repeat(1, 1, pad_length)
            segment = torch.cat((segment, pad_values), dim=2)

            # 将切分后的段赋值给结果张量
        result[:, :, i, :] = segment

    return result
class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.h_channel = configs.h_channel
            self.atten_bias=configs.atten_bias
            self.global_bias = configs.global_bias
            self.local_bias = configs.local_bias
            self.linear_channel_out = nn.Linear(self.h_channel, configs.enc_in, bias=True)
            self.linear_channel_in = nn.Linear(configs.enc_in, self.h_channel, bias=True)
            self.norm_channel = nn.BatchNorm1d(self.h_channel)
            self.ff = nn.Sequential(nn.GELU(),
                                    nn.Dropout(configs.fc_dropout))
            self.dropout=nn.Dropout(configs.fc_dropout)
            decoder_cross_att = ProbAttention()
            self.decoder_channel = AttentionLayer1(
                decoder_cross_att,
                configs.enc_in, configs.n_heads)
            self.decoder_res = AttentionLayer1(
                decoder_cross_att,
                configs.pred_len, configs.n_heads)
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone_decom(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone_decom(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=16, stride=8,
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res2 = PatchTST_backbone_decom(c_in=c_in, context_window=context_window,
                                                     target_window=target_window, patch_len=48, stride=6,
                                                     max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                     n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout,
                                                     dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                     padding_var=padding_var,
                                                     attn_mask=attn_mask, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn,
                                                     pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                     head_dropout=head_dropout, padding_patch=padding_patch,
                                                     pretrain_head=pretrain_head, head_type=head_type,
                                                     individual=individual, revin=revin, affine=affine,
                                                     subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res3 = PatchTST_backbone_decom(c_in=c_in, context_window=context_window,
                                                     target_window=target_window, patch_len=30, stride=6,
                                                     max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                     n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout,
                                                     dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                     padding_var=padding_var,
                                                     attn_mask=attn_mask, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn,
                                                     pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                     head_dropout=head_dropout, padding_patch=padding_patch,
                                                     pretrain_head=pretrain_head, head_type=head_type,
                                                     individual=individual, revin=revin, affine=affine,
                                                     subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.revin=configs.revin
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
            self.lstm = nn.GRU(input_size=1, hidden_size=6, num_layers=1, batch_first=True)
            self.Linear = nn.Linear(configs.enc_in//1 * 6
                                    , configs.enc_in)
            self.lstm_time = nn.GRU(input_size=6, hidden_size=4, num_layers=1, batch_first=True)
            self.Linear_time = nn.Linear(configs.seq_len//6 * 4
                                    , configs.pred_len)
            self.lstm_res = nn.GRU(input_size=24, hidden_size=48, num_layers=1, batch_first=True)
            self.Linear_res = nn.Linear(configs.pred_len*3//24*48
                                    , configs.pred_len)
            self.Linear1= nn.Linear(configs.seq_len
                                    , configs.pred_len)
            self.pred_len =configs.pred_len
            self.enc_in = configs.enc_in
            self.norm_channel = nn.BatchNorm1d(self.h_channel)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            if self.revin:
                x = self.revin_layer(x, 'norm')
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            res2 = self.model_res2(res_init)
            res3 = self.model_res3(res_init)
            # trend = self.model_trend(trend_init).permute(0,2,1)
            res_new=torch.cat([res,res2,res3],dim=-1,out=None)
            res_new = split_and_pad_sequence(res_new, 24, 24)
            res_new = res_new.reshape(-1, res_new.shape[-2], res_new.shape[-1]).to(trend_init.device)
            res_all, res_new = self.lstm_res(res_new)

            res_all = res_all.reshape(res_all.shape[0], -1)
            res_all = self.Linear_res(res_all)

            res_all = res_all.reshape(x.shape[0], x.shape[-1], -1).permute(0, 2, 1)
#gru-channel
            trend_init=self.Linear1(trend_init)
            trend_init=trend_init.permute(0,2,1)#刚加的
            data = split_and_pad_sequence(trend_init, 1, 1)
            data = data.reshape(-1, data.shape[-2], data.shape[-1]).to(trend_init.device)
            rnn_output, _ = self.lstm(data)
            rnn_output = rnn_output.reshape(rnn_output.shape[0], -1)
            rnn_output = self.Linear(rnn_output)
            rnn_output = rnn_output.reshape(x.shape[0], self.pred_len,self.enc_in)
# #gru-time
#             data = split_and_pad_sequence(trend_init, 6, 6)
#             data = data.reshape(-1, data.shape[-2], data.shape[-1]).to(trend_init.device)
#             rnn_output, _ = self.lstm_time(data)
#             rnn_output = rnn_output.reshape(rnn_output.shape[0], -1)
#             rnn_output = self.Linear_time(rnn_output)
#             rnn_output = rnn_output.reshape(x.shape[0], x.shape[-1], -1).permute(0, 2, 1)

            # output = rnn_output + res_all + res.permute(0,2,1) +res2.permute(0,2,1) + res3.permute(0,2,1)
            output = rnn_output + res_all
            if self.revin:
                output = self.revin_layer(output, 'denorm')

        else:
            res=x
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            output = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return output,res