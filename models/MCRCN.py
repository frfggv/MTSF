
# Cell
from typing import Callable, Optional
import torch
from sklearn.cluster import KMeans
from torch import nn
from collections import defaultdict
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN,RevIN_trend
from layers.PatchTST_backbone import PatchTST_backbone_decom,PatchTST_backbone_gru,PatchTST_backbone_gru2,PatchTST_backbone_gru3
from layers.PatchTST_backbone import PatchTST_backbone_gru4,PatchTST_backbone_gru5,PatchTST_backbone_gru6,PatchTST_backbone_gru7,PatchTST_backbone_gru8,TSTiEncoder
from layers.PatchTST_backbone import Flatten_Head, PatchTST_backbone,res1_reshape,Linear_flatten,res_MCCNreshape
from layers.PatchTST_layers import series_decomp
from layers.global_conv import GConv
from layers.SelfAttention_Family import AttentionLayer1, FullAttention, ProbAttention
from models import Autoformer, DLinear, Informer
from layers.TCN import TemporalConvNet
from layers.Embed import TimeFeatureEmbedding,DataEmbedding1
from mamba_ssm import Mamba
from einops import rearrange
from layers.mamba_layer import Encoder, EncoderLayer, EncoderLayer2, EncoderLayer3
seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
class convbackbone(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(
                configs.batch_size,
                configs.batch_size,
                # kernel_size=(2,2),
                kernel_size=(4, 4),
                stride=1,
                padding='same'
            ),
            nn.GELU()
        )

    def forward(self, x):
        x=self.conv(x)+x
        return x
class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P - self.S),
            mode='replicate'
        )  # [B*M, 1, L] -> [B*M, 1, L+P-S]

        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]

        return x_emb  # x_emb: [B, M, D, N]


class ConvFFN(nn.Module):
    def __init__(self, D, r, kernel, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        # groups_num = M if one else D

        self.pw_con1 = nn.Conv2d(
            in_channels= D,
            out_channels=r * D,
            kernel_size=kernel,

            stride=1,
            padding='same'
            # groups=groups_num
        )
        self.pw_con2 = nn.Conv2d(
            in_channels=r  * D,
            out_channels= D,
            kernel_size=kernel,

            stride=1,
            padding='same'
            # groups=groups_num
        )

    def forward(self, x):
        # x: [B, D*M, N]
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
        return x  # x: [B, D*M, N]


class V_T_Conv_Block(nn.Module):
    def __init__(self, N, D, kernel_size,conv2d_kernel, conv2d_kernel2, r):
        super(V_T_Conv_Block, self).__init__()
        #深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=N * D,
            out_channels=N * D,
            kernel_size=kernel_size,
            groups=N * D,
            padding='same'
        )

        self.conv_ffn1 = ConvFFN(D, r, (4,conv2d_kernel), one=True)
        self.conv_ffn2 = ConvFFN(D, r, (4,conv2d_kernel2), one=False)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]


        x=x_emb.permute(0,3,2,1)# [B, N, D, M]
        x = rearrange(x, 'b m d n -> b (m d) n')  # [B, N, D, M] -> [B, N*D, M]

        x = self.dw_conv(x)  # [B, N*D, M] -> [B, N*D, M]
        x = rearrange(x, 'b (m d) n -> b m d n', d=D)
        x = x.permute(0, 2, 3, 1)#[B, D, M, N]
        # x = self.bn(x)  # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_ffn1(x)  # [B, D, M, N] -> [B, D,M, N]

        # x = rearrange(x, 'b (m d) n -> b m d n', d=D)  # [B, M*D, N] -> [B, M, D, N]
        # x = x.permute(0, 2, 1, 3)  # [B, M, D, N] -> [B, D, M, N]
        # x = rearrange(x, 'b d m n -> b (d m) n')  # [B, D, M, N] -> [B, D*M, N]

        # x = self.conv_ffn2(x)  # [B, D, M, N] -> [B, D,M, N]

        # x = rearrange(x, 'b (d m) n -> b d m n', d=D)  # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0, 2, 1, 3)  # [B, D, M, N] -> [B, M, D, N]
        out = x + x_emb

        return out  # out: [B, M, D, N]


class V_T_Conv(nn.Module):
    def __init__(self, conv2d_kernel,conv2d_kernel2, M, L, T, d_model, D=20, P=8, S=4, kernel_size=4, r=1, num_layers=2):
        super(V_T_Conv, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D)
        self.backbone = nn.ModuleList([V_T_Conv_Block(N, D, kernel_size, conv2d_kernel,conv2d_kernel2, r) for _ in range(num_layers)])
        self.head = nn.Linear(D * N, T)

    def forward(self, x):
        # x: [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]

        # Flatten
        z = rearrange(x_emb, 'b m d n -> b m (d n)')  # [B, M, D, N] -> [B, M, D*N]
        pred = self.head(z)  # [B, M, D*N] -> [B, M, T]

        return pred  # out: [B, M, T]

class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.d_model = configs.d_model
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        gru_model=configs.gru_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patchlen1
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition

        # model
        self.decomposition = decomposition
        self.enc_in = configs.enc_in
        if self.decomposition:
            self.gru_hidden=configs.gru_hidden
            # self.conv2d = nn.Conv2d(in_channels=configs.top_k, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.valueEmbedding = nn.Sequential(
                nn.Linear(configs.seq_len, configs.d_model),
                nn.ReLU()
            )
            self.predict = nn.Sequential(
                nn.Linear(configs.d_model, self.enc_in)
            )
            self.patchlen1=configs.patchlen1
            self.patchlen2=configs.patchlen2
            self.patchlen3=configs.patchlen3
            self.k=configs.top_k
            self.u=configs.u
            self.u_seasonal=configs.u_seasonal
            self.u_trend=configs.u_trend
            self.linear_seq_pred = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
            self.h_channel = configs.h_channel
            self.atten_bias = configs.atten_bias
            self.global_bias = configs.global_bias
            self.local_bias = configs.local_bias
            self.linear_channel_out = nn.Linear(self.k, 1)
            self.linear_channel_in = nn.Linear(configs.enc_in, self.h_channel, bias=True)
            self.norm_channel = nn.BatchNorm1d(self.h_channel)
            self.ff = nn.Sequential(nn.GELU(),
                                    nn.Dropout(configs.fc_dropout))
            self.dropout = nn.Dropout(configs.dropout)
            self.dropout_conv = nn.Dropout(configs.conv_dropout)
            self.dropout_seasonal=nn.Dropout(0.1)
            self.dropout_trend = nn.Dropout(0.2)
            decoder_cross_att = ProbAttention()
            self.decoder_channel = AttentionLayer1(
                decoder_cross_att,
                configs.enc_in, configs.n_heads)
            self.decoder_res = AttentionLayer1(
                decoder_cross_att,
                configs.pred_len, configs.n_heads)
            self.decomp_module = series_decomp(self.patchlen1)
            self.decomp_module2 = series_decomp(self.patchlen2)
            self.decomp_module3 = series_decomp(self.patchlen3)
            self.decomp_trend = series_decomp(configs.gru_model//8)
            self.batch_size=configs.batch_size
            self.seq_len=configs.seq_len
            self.pred_len = configs.pred_len

            self.global_layer_Gconv = GConv(configs.batch_size, d_model=configs.enc_in, d_state=configs.enc_in,
            l_max=configs.seq_len, channels=configs.n_heads, bidirectional=True, kernel_dim=32, n_scales=None, decay_min=2, decay_max=2, transposed=False)
            self.model_trend = PatchTST_backbone_decom(c_in=c_in, context_window=context_window,
                                                       target_window=target_window, patch_len=patch_len, stride=stride,
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
            self.model_res = res_MCCNreshape(hidden=self.gru_hidden, batch_size=self.batch_size,
                                          pred_len=self.pred_len, c_in=c_in, context_window=context_window,
                                          target_window=target_window, patch_len=configs.patchlen1,
                                          stride=configs.patchlen1,
                                          max_seq_len=max_seq_len, n_layers=n_layers, d_model=self.d_model,
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
            self.model_res2 = res1_reshape(hidden=self.gru_hidden, batch_size=self.batch_size,
                                           pred_len=self.pred_len, c_in=c_in, context_window=context_window,
                                           target_window=target_window, patch_len=configs.patchlen2//self.u_seasonal,
                                           stride=configs.patchlen2 // 3,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=configs.gru_model,
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
            self.model_res3 = res1_reshape(hidden=self.gru_hidden, batch_size=self.batch_size,
                                           pred_len=self.pred_len, c_in=c_in, context_window=context_window,
                                           target_window=target_window, patch_len=configs.patchlen3//self.u_seasonal,
                                           stride=configs.patchlen3 // 3,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=configs.gru_model,
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
            self.flatten1 = Linear_flatten(hidden=configs.gru_model, batch_size=self.batch_size,
                                          pred_len=self.pred_len, c_in=c_in, context_window=context_window,
                                          target_window=target_window, patch_len=configs.patchlen1//self.u_seasonal,
                                          stride=configs.patchlen1 // 3,
                                          max_seq_len=max_seq_len, n_layers=n_layers, d_model=configs.gru_model,
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
            self.flatten2 = Linear_flatten(hidden=configs.gru_model, batch_size=self.batch_size,
                                           pred_len=self.pred_len, c_in=c_in, context_window=context_window,
                                           target_window=target_window, patch_len=configs.patchlen2//self.u_seasonal,
                                           stride=configs.patchlen2 // 3,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=configs.gru_model,
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
            self.flatten3 = Linear_flatten(hidden=configs.gru_model, batch_size=self.batch_size,
                                           pred_len=self.pred_len, c_in=c_in, context_window=context_window,
                                           target_window=target_window, patch_len=configs.patchlen3//self.u_seasonal,
                                           stride=configs.patchlen3 // 3,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=configs.gru_model,
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
            self.TCN = TemporalConvNet(configs.enc_in, [configs.h_channel, configs.enc_in])
            self.TCN1=TemporalConvNet(configs.seq_len, [configs.seq_len*2, configs.pred_len])
            self.local_Ajiutoformer = Autoformer.Model(configs)
            self.local_DLinear = DLinear.Model(configs)

            self.local_Informer=Informer.Model(configs)
            self.res_model=configs.res_model
            self.trend_model = configs.trend_model
            self.trend_only =configs.trend_only
            self.seasonal_only = configs.seasonal_only
            self.gru=nn.GRU(input_size=self.enc_in, hidden_size=configs.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
            self.merge = torch.nn.Conv2d(in_channels=self.enc_in, out_channels=self.enc_in,
                                         kernel_size=(self.k, 1))
            self.merge_1d = torch.nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                         kernel_size=self.k,stride=self.k)
            # self.merge_1d = torch.nn.Conv1d(in_channels=1 * self.pred_len,
            #                                 out_channels=self.enc_in * self.pred_len,
            #                                 kernel_size=self.k)
            self.merge_2d_new = torch.nn.Conv2d(in_channels=self.enc_in,
                                            out_channels=self.enc_in,
                                            kernel_size=(1,self.k))
            self.merge_2d = torch.nn.Conv2d(in_channels=self.k,
                                            out_channels=self.k,
                                            kernel_size=(1,1))
            self.merge_type=configs.merge
            self.weight1=configs.weight1
            self.weight2 = configs.weight2
            self.weight3 = configs.weight3
            self.weight4 = configs.weight4
            self.weight5 = configs.weight5
            self.lstm_trend = nn.GRU(input_size=configs.gru_model, hidden_size=configs.gru_model, num_layers=1, batch_first=True)
            self.lstm_trend2 = nn.GRU(input_size=configs.gru_model, hidden_size=configs.gru_model, num_layers=1,
                                     batch_first=True)
            self.lstm_trend3 = nn.GRU(input_size=configs.gru_model, hidden_size=configs.gru_model, num_layers=1,
                                     batch_first=True)

            self.gru_seasonal = nn.GRU(input_size=d_model, hidden_size=self.gru_hidden, num_layers=1, batch_first=True)
            self.revin = configs.revin
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
            self.revin_trend = RevIN_trend(c_in, affine=affine, subtract_last=subtract_last)
            if padding_patch == 'end':  # can be modified to general case
                self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            # self.lstm_trend = nn.GRU(input_size=6, hidden_size=4, num_layers=1, batch_first=True)
            # self.Linear_trend = nn.Linear((configs.seq_len) // 6 * 4
            #                              , configs.pred_len)
            # self.lstm = nn.GRU(input_size=1, hidden_size=6, num_layers=1, batch_first=True)
            # self.Linear = nn.Linear(configs.enc_in // 1 * 6
            #                         , configs.enc_in)
            # self.lstm_time = nn.GRU(input_size=6, hidden_size=12, num_layers=1, batch_first=True)
            # self.Linear_time = nn.Linear(( configs.seq_len+configs.pred_len)// 6 * 12
            #                              , configs.pred_len)
            # self.lstm_res = nn.GRU(input_size=24, hidden_size=48, num_layers=1, batch_first=True)
            # self.Linear_res = nn.Linear(configs.pred_len * configs.top_k
            #                             , configs.pred_len)
            if context_window//configs.patchlen1==0:
                self.patch_num = int((context_window - configs.patchlen1) // patch_len + 1)
            else:self.patch_num=int((context_window - configs.patchlen1) // patch_len + 2)

            self.Linear1 = nn.Linear(self.seq_len
                                     , self.pred_len)
            self.Linear2 = nn.Linear(self.seq_len
                                     , self.pred_len)
            self.L=configs.L
            self.Linear_fusion = nn.Linear(self.enc_in
                                     , 1)
            self.Linear_fusionback = nn.Linear(d_model
                                           , self.enc_in)
            self.embed=TimeFeatureEmbedding(d_model=d_model, embed_type='timeF', freq=configs.freq)
            self.embed_new=TimeFeatureEmbedding(d_model=self.enc_in, embed_type='timeF', freq=configs.freq)
            self.DataEmbedding1=DataEmbedding1(c_in=self.enc_in, d_model=d_model)
            self.norm_channel = nn.BatchNorm1d(self.h_channel)
            self.att_model=self.pred_len
            self.attention = TSTiEncoder(c_in, patch_num=2, patch_len=self.pred_len, max_seq_len=max_seq_len,
                                        n_layers=n_layers, d_model=self.att_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
                                        d_ff=d_ff,
                                        attn_dropout=attn_dropout, dropout=dropout, act=act,
                                        key_padding_mask=key_padding_mask, padding_var=padding_var,
                                        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                        store_attn=store_attn,
                                        pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
            self.head_nf = self.att_model * 2
            self.n_vars = c_in
            # self.attn = nn.Sequential(
            #     nn.Linear(configs.d_model*3, configs.d_ff),
            #     nn.Tanh(),
            #     nn.Linear(configs.d_ff, self.k),
            # )
            self.attn = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.Tanh(),
                nn.Linear(configs.d_ff, self.k),
            )
            self.trend_Linear = nn.Sequential(
                nn.Linear(configs.gru_hidden, configs.gru_hidden),
                nn.Tanh(),
                nn.Linear(configs.gru_hidden, configs.gru_hidden),
                nn.Dropout(configs.dropout)
            )
            self.individual = individual
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
            self.W_P = nn.Linear(self.patchlen1//self.u_trend, configs.gru_model)
            self.W_P2 = nn.Linear(self.patchlen2//self.u_trend, configs.gru_model)
            self.W_P3 = nn.Linear(self.patchlen3//self.u_trend, configs.gru_model)
        #     self.convblock=nn.ModuleList([
        #       convbackbone(configs)
        #     for i in range(configs.e_layers)
        # ])
            self.Conv2d = V_T_Conv(configs.conv2d_kernel,configs.conv2d_kernel2, self.batch_size, self.seq_len, self.seq_len,self.d_model,D=configs.dim,
                                    kernel_size=configs.kernel_size, r=configs.r, num_layers=configs.num_layer)
            self.Conv2d_res = V_T_Conv(configs.conv2d_kernel, configs.conv2d_kernel2, self.batch_size, self.seq_len, self.seq_len, self.d_model,D=configs.dim,
                                        kernel_size=configs.kernel_size, r=configs.r, num_layers=configs.num_layer)
            self.layer=configs.e_layers
            self.layer_norm=nn.LayerNorm(self.d_model)
            self.encoder1 = Encoder(
                [
                    EncoderLayer(
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        self.seq_len,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.mam_layer)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)#可以为None
            )
            self.encoder2 = Encoder(
                [
                    EncoderLayer2(
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        self.seq_len,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.mam_layer)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)  # 可以为None
            )
            self.encoder3 = Encoder(
                [
                    EncoderLayer3(
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        self.seq_len,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.mam_layer)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)  # 可以为None
            )
            self.encoder1_res = Encoder(
                [
                    EncoderLayer(
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        self.seq_len,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.mam_layer)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)  # 可以为None
            )
            self.encoder2_res = Encoder(
                [
                    EncoderLayer2(
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        self.seq_len,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.mam_layer)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)  # 可以为None
            )
            self.encoder3_res = Encoder(
                [
                    EncoderLayer3(
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=self.seq_len,  # Model dimension d_model
                            # d_model=7,
                            d_state=self.seq_len,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        self.seq_len,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.mam_layer)
                ],
                norm_layer=torch.nn.LayerNorm(self.seq_len)  # 可以为None
            )
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

            self.mam_model=self.seq_len
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            if self.revin:
                x = self.revin_layer(x, 'norm')
            res_init, trend_init = self.decomp_module(x)
            res_init,trend_init=res_init.permute(0,2,1), trend_init.permute(0,2,1)
            # res_init=self.model_res(res_init)
            #trend process
            # trend_init=self.model_res(trend_init) # x: x: [bs x (patch_num * nvars) x d_model]
            trend_mamba1, _ =self.encoder1(trend_init)
            trend_mamba2, _=self.encoder2(trend_init)

            c=trend_init.shape[-2]
            random_indices = torch.randperm(c)
            # 根据随机索引重排数据
            shuffled_trend_init = trend_init[:, random_indices, :]

            # 保存原始索引和新索引
            original_to_new_indices = {i: random_indices[i].item() for i in range(c)}
            new_to_original_indices = {v: k for k, v in original_to_new_indices.items()}


            #随机重排Mamba
            trend_mamba3, _ = self.encoder3(shuffled_trend_init)
            # 步骤1: 创建逆置换张量
            inverse_indices = torch.zeros_like(random_indices)
            for new_idx, original_idx in new_to_original_indices.items():
                inverse_indices[new_idx] = original_idx

            # 步骤2: 使用逆置换重新排序，将顺序恢复到原始状态
            restored_trend_mamba3 = trend_mamba3[:, inverse_indices, :]+trend_mamba1


            # Step 1: 对每个 batch 的 n 个 d 维向量求平均值，得到 n 个平均向量
            mean_vectors = trend_init.mean(dim=0)  # mean_vectors 的形状为 (n, d)
            mean_vectors=mean_vectors.detach().cpu()
            # Step 2: 对这些平均向量进行 K-means 聚类
            kmeans = KMeans(n_clusters=self.k, random_state=0).fit(mean_vectors)
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_  # 得到每个 n 个平均向量的聚类标签

            # Step 3: 将聚类标签应用到每个 batch
            # labels_reshaped = torch.empty((self.batch_size, trend_mamba1.shape[1]), dtype=torch.long)
            # for i in range(self.batch_size):
            #     labels_reshaped[i] = torch.tensor(labels)
            labels_reshaped = torch.tensor(labels, dtype=torch.long).repeat(self.batch_size, 1)

            b=self.batch_size
            n=trend_mamba1.shape[1]
            k=self.k
            d=trend_mamba1.shape[-1]
            #以下为求序列到聚类中心的概率分布
            t_value = 1.0  # 参数 t，可以根据实际情况调整
            cluster_centers = torch.from_numpy(cluster_centers)
            # Step 1: 扩展聚类中心的维度
            expanded_cluster_centers = cluster_centers.unsqueeze(0).unsqueeze(0).to(trend_mamba1.device)  # 形状变为 (1, 1, k, t)

            # Step 2: 计算时间序列与每个聚类中心之间的差值的平方
            diff_squared = (trend_mamba1.unsqueeze(2) - expanded_cluster_centers) ** 2  # 形状为 (b, c, k, t)

            # 在时间维度上求和，得到每个序列到每个聚类中心的距离平方
            distance_squared = diff_squared.sum(dim=-1)  # 形状为 (b, c, k)

            # Step 3: 计算 q_{ij} 的分子
            numerator = (1 + distance_squared / t_value) ** (-(t_value + 1) / 2)  # 形状为 (b, c, k)

            # Step 4: 计算 q_{ij} 的分母
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 5: 计算最终的 q_{ij}
            q_ij = numerator / denominator  # 形状为 (b, c, k)
            q_ij=q_ij.to(trend_mamba1.device)
            # Step 1: 计算 f_j for each batch
            # 先对 c 维度求和，然后保持 batch 维度和聚类维度
            f_j = q_ij.sum(dim=1)  # 形状为 (b, k)

            # Step 2: 计算分子部分
            # 需要扩展 f_j 维度以便进行广播
            numerator = (q_ij ** 2) / f_j.unsqueeze(1)  # f_j 形状扩展为 (b, 1, k)，numerator 形状为 (b, c, k)

            # Step 3: 计算分母部分
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 4: 计算最终的 p_ij
            p_ij_trend_mam1 = numerator / denominator  # 形状为 (b, c, k)

            #以下为求序列到聚类中心的概率分布
            t_value = 1.0  # 参数 t，可以根据实际情况调整

            # Step 1: 扩展聚类中心的维度
            expanded_cluster_centers = cluster_centers.unsqueeze(0).unsqueeze(0).to(trend_mamba1.device)  # 形状变为 (1, 1, k, t)

            # Step 2: 计算时间序列与每个聚类中心之间的差值的平方
            diff_squared = (restored_trend_mamba3.unsqueeze(2) - expanded_cluster_centers) ** 2  # 形状为 (b, c, k, t)

            # 在时间维度上求和，得到每个序列到每个聚类中心的距离平方
            distance_squared = diff_squared.sum(dim=-1)  # 形状为 (b, c, k)

            # Step 3: 计算 q_{ij} 的分子
            numerator = (1 + distance_squared / t_value) ** (-(t_value + 1) / 2)  # 形状为 (b, c, k)

            # Step 4: 计算 q_{ij} 的分母
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 5: 计算最终的 q_{ij}
            q_ij = numerator / denominator  # 形状为 (b, c, k)
            q_ij=q_ij.to(trend_mamba1.device)
            # Step 1: 计算 f_j for each batch
            # 先对 c 维度求和，然后保持 batch 维度和聚类维度
            f_j = q_ij.sum(dim=1)  # 形状为 (b, k)

            # Step 2: 计算分子部分
            # 需要扩展 f_j 维度以便进行广播
            numerator = (q_ij ** 2) / f_j.unsqueeze(1)  # f_j 形状扩展为 (b, 1, k)，numerator 形状为 (b, c, k)

            # Step 3: 计算分母部分
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 4: 计算最终的 p_ij
            p_ij_trend_mam3 = numerator / denominator  # 形状为 (b, c, k)

            cluster_tensors_trend = [None] * self.k

            for cluster_id in range(self.k):
                # 找出所有属于当前聚类的序列索引 (取第一个batch的标签即可，因为标签相同)
                cluster_indices = (labels_reshaped[0] == cluster_id).nonzero(as_tuple=True)[0]

                if len(cluster_indices) > 0:
                    # 对所有 batch 一次性提取属于当前聚类的序列
                    batch_tensors = trend_mamba1[:, cluster_indices, :]

                    # 将每个 batch 中的序列堆叠在一起
                    cluster_tensors_trend[cluster_id] = batch_tensors


            for k in range(self.k):
                cluster_tensors_trend[k]=cluster_tensors_trend[k].to(trend_init.device)


                    # cluster_tensors_trend[k] = (self.convblock[i](cluster_tensors_trend[k]))
                cluster_tensors_trend[k] = self.Conv2d(cluster_tensors_trend[k])


            # reconstructed = torch.zeros_like(trend_mamba1)
            # # 将每个聚类的序列放回原始位置
            # for cluster_id in range(self.k):
            #     for i in range(self.batch_size):
            #         # 找到属于当前聚类的序列的索引
            #         batch_indices = (labels_reshaped[i] == cluster_id).nonzero(as_tuple=True)[0]
            #         if len(batch_indices) > 0:
            #             # 将当前聚类的张量填充到重建张量的相应位置
            #             num_sequences = len(batch_indices)
            #             reconstructed[i, batch_indices] = cluster_tensors_trend[cluster_id][i, :num_sequences]
            reconstructed = torch.zeros_like(trend_mamba1)

            for cluster_id in range(self.k):
                # 找到属于当前聚类的序列的索引（一次性处理所有batch）
                batch_indices = (labels_reshaped[0] == cluster_id).nonzero(as_tuple=True)[0]

                if len(batch_indices) > 0:
                    # 提取并重建属于该聚类的所有 batch 的序列
                    cluster_data = cluster_tensors_trend[cluster_id][:, :len(batch_indices), :]
                    reconstructed[:, batch_indices, :] = cluster_data

            trend=reconstructed.reshape(reconstructed.shape[0],-1,self.mam_model*reconstructed.shape[1]//self.enc_in)

            trend = self.Linear1(trend)

            #seasonal process
            # res_init=self.model_res(res_init)# x: x: [bs x (patch_num * nvars) x d_model]

            res_mamba1, _ = self.encoder1_res(res_init)
            res_mamba2, _ = self.encoder2_res(res_init)

            c = res_init.shape[-2]
            random_indices = torch.randperm(c)
            # 根据随机索引重排数据
            shuffled_res_init = res_init[:, random_indices, :]

            # 保存原始索引和新索引
            original_to_new_indices = {i: random_indices[i].item() for i in range(c)}
            new_to_original_indices = {v: k for k, v in original_to_new_indices.items()}

            # 随机重排Mamba
            res_mamba3, _ = self.encoder3_res(shuffled_res_init)
            # 步骤1: 创建逆置换张量
            inverse_indices = torch.zeros_like(random_indices)
            for new_idx, original_idx in new_to_original_indices.items():
                inverse_indices[new_idx] = original_idx

            # 步骤2: 使用逆置换重新排序，将顺序恢复到原始状态
            restored_res_mamba3 = res_init[:, inverse_indices, :]+res_mamba1

            # Step 1: 对每个 batch 的 n 个 d 维向量求平均值，得到 n 个平均向量
            mean_vectors = res_init.mean(dim=0)  # mean_vectors 的形状为 (n, d)
            mean_vectors = mean_vectors.detach().cpu()
            # Step 2: 对这些平均向量进行 K-means 聚类
            kmeans = KMeans(n_clusters=self.k, random_state=0).fit(mean_vectors)
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_  # 得到每个 n 个平均向量的聚类标签

            # Step 3: 将聚类标签应用到每个 batch
            labels_reshaped = torch.tensor(labels, dtype=torch.long).repeat(self.batch_size, 1)

            b = self.batch_size
            n = res_mamba1.shape[1]
            k = self.k
            d = res_mamba1.shape[-1]
            # 以下为求序列到聚类中心的概率分布
            t_value = 1.0  # 参数 t，可以根据实际情况调整
            cluster_centers = torch.from_numpy(cluster_centers)
            # Step 1: 扩展聚类中心的维度
            expanded_cluster_centers = cluster_centers.unsqueeze(0).unsqueeze(0).to(
                trend_mamba1.device)  # 形状变为 (1, 1, k, t)

            # Step 2: 计算时间序列与每个聚类中心之间的差值的平方
            diff_squared = (res_mamba1.unsqueeze(2) - expanded_cluster_centers) ** 2  # 形状为 (b, c, k, t)

            # 在时间维度上求和，得到每个序列到每个聚类中心的距离平方
            distance_squared = diff_squared.sum(dim=-1)  # 形状为 (b, c, k)

            # Step 3: 计算 q_{ij} 的分子
            numerator = (1 + distance_squared / t_value) ** (-(t_value + 1) / 2)  # 形状为 (b, c, k)

            # Step 4: 计算 q_{ij} 的分母
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 5: 计算最终的 q_{ij}
            q_ij = numerator / denominator  # 形状为 (b, c, k)
            q_ij = q_ij.to(trend_mamba1.device)
            # Step 1: 计算 f_j for each batch
            # 先对 c 维度求和，然后保持 batch 维度和聚类维度
            f_j = q_ij.sum(dim=1)  # 形状为 (b, k)

            # Step 2: 计算分子部分
            # 需要扩展 f_j 维度以便进行广播
            numerator = (q_ij ** 2) / f_j.unsqueeze(1)  # f_j 形状扩展为 (b, 1, k)，numerator 形状为 (b, c, k)

            # Step 3: 计算分母部分
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 4: 计算最终的 p_ij
            p_ij_res_mam1 = numerator / denominator  # 形状为 (b, c, k)

            # 以下为求序列到聚类中心的概率分布
            t_value = 1.0  # 参数 t，可以根据实际情况调整

            # Step 1: 扩展聚类中心的维度
            expanded_cluster_centers = cluster_centers.unsqueeze(0).unsqueeze(0).to(
                res_mamba1.device)  # 形状变为 (1, 1, k, t)

            # Step 2: 计算时间序列与每个聚类中心之间的差值的平方
            diff_squared = (restored_res_mamba3.unsqueeze(2) - expanded_cluster_centers) ** 2  # 形状为 (b, c, k, t)

            # 在时间维度上求和，得到每个序列到每个聚类中心的距离平方
            distance_squared = diff_squared.sum(dim=-1)  # 形状为 (b, c, k)

            # Step 3: 计算 q_{ij} 的分子
            numerator = (1 + distance_squared / t_value) ** (-(t_value + 1) / 2)  # 形状为 (b, c, k)

            # Step 4: 计算 q_{ij} 的分母
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 5: 计算最终的 q_{ij}
            q_ij = numerator / denominator  # 形状为 (b, c, k)
            q_ij = q_ij.to(res_mamba1.device)
            # Step 1: 计算 f_j for each batch
            # 先对 c 维度求和，然后保持 batch 维度和聚类维度
            f_j = q_ij.sum(dim=1)  # 形状为 (b, k)

            # Step 2: 计算分子部分
            # 需要扩展 f_j 维度以便进行广播
            numerator = (q_ij ** 2) / f_j.unsqueeze(1)  # f_j 形状扩展为 (b, 1, k)，numerator 形状为 (b, c, k)

            # Step 3: 计算分母部分
            denominator = numerator.sum(dim=-1, keepdim=True)  # 在 k 维度上求和，形状为 (b, c, 1)

            # Step 4: 计算最终的 p_ij
            p_ij_res_mam3 = numerator / denominator  # 形状为 (b, c, k)

            # 初始化每个聚类的张量，形状为 [b, n', d]
            cluster_tensors_res = [None] * self.k
            for cluster_id in range(self.k):
                # 找出所有属于当前聚类的序列索引 (取第一个batch的标签即可，因为标签相同)
                cluster_indices = (labels_reshaped[0] == cluster_id).nonzero(as_tuple=True)[0]

                if len(cluster_indices) > 0:
                    # 对所有 batch 一次性提取属于当前聚类的序列
                    batch_tensors = res_mamba1[:, cluster_indices, :]

                    # 将每个 batch 中的序列堆叠在一起
                    cluster_tensors_res[cluster_id] = batch_tensors


            for k in range(self.k):
                cluster_tensors_res[k] = cluster_tensors_res[k].to(res_init.device)

                    # cluster_tensors_res[k] = self.layer_norm(self.convblock[i](cluster_tensors_res[k]))
                cluster_tensors_res[k] = self.Conv2d_res(cluster_tensors_res[k])
            reconstructed = torch.zeros_like(res_mamba1)

            for cluster_id in range(self.k):
                # 找到属于当前聚类的序列的索引（一次性处理所有batch）
                batch_indices = (labels_reshaped[0] == cluster_id).nonzero(as_tuple=True)[0]

                if len(batch_indices) > 0:
                    # 提取并重建属于该聚类的所有 batch 的序列
                    cluster_data = cluster_tensors_res[cluster_id][:, :len(batch_indices), :]
                    reconstructed[:, batch_indices, :] = cluster_data

            res = reconstructed.reshape(reconstructed.shape[0], -1,
                                          self.mam_model * reconstructed.shape[1] // self.enc_in)

            res = self.Linear2(res)


            out=self.dropout(trend+res).permute(0,2,1)

            out = self.revin_layer(out, 'denorm')

        return trend_mamba1,trend_mamba2,restored_trend_mamba3,res_mamba1,res_mamba2,restored_res_mamba3,p_ij_trend_mam1, p_ij_trend_mam3, p_ij_res_mam1,p_ij_res_mam3, out
        # return output, mg,trend