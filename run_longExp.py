import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--res_model', type=str, required=False, default='msgru', help='status')
parser.add_argument('--trend_model', type=str, required=False, default='patchtst', help='status')
parser.add_argument('--trend_only', type=bool, required=False, default=False, help='status')
parser.add_argument('--seasonal_only', type=bool, required=False, default=False, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--rnn', type=str, required=False, default='gru', help='model id')
# data loader
parser.add_argument('--data', type=str, required=True, default='traffic', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/traffic/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')


# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# TimesNet
parser.add_argument('--num_kernels', type=int, default=3, help='merge')
parser.add_argument('--num_clusters', type=int, default=3, help='merge')
# PatchTST
parser.add_argument('--merge', type=str, default='time-aware', help='merge')
parser.add_argument('--seasonal_dropout', type=float, default=0.2, help='fully connected dropout')
parser.add_argument('--fc_dropout', type=float, default=0.2, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')

parser.add_argument('--kernel_size2', type=int, default=48, help='decomposition-kernel')
parser.add_argument('--kernel_size3', type=int, default=32, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
#GCformer
parser.add_argument('--L', type=int, default=1, help='')
parser.add_argument('--h_channel', type=int, default=32, help='')
parser.add_argument('--u_trend', type=int, default=2, help='')
parser.add_argument('--u_seasonal', type=int, default=1, help='')
parser.add_argument('--u', type=int, default=2, help='')
parser.add_argument('--atten_bias', type=float, default=0.5)
parser.add_argument('--local_bias', type=float, default=0.5, help='pred = pred + local_bias*local_output')
parser.add_argument('--global_bias', type=float, default=0.5, help='pred = pred + global_bias*global_output ')
# Formers 
parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=7, help='dimension of model')
parser.add_argument('--gru_model', type=int, default=24, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=24, help='window size of moving average')
parser.add_argument('--moving_avg1', type=int, default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

parser.add_argument('--conv_dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--gru_hidden', type=int, default=24, help='k seasonals')
parser.add_argument('--patchlen1', type=int, default=24, help='k seasonals')
parser.add_argument('--patchlen2', type=int, default=16, help='k seasonals')
parser.add_argument('--patchlen3', type=int, default=12, help='k seasonals')
parser.add_argument('--patchlen4', type=int, default=24, help='k seasonals')
parser.add_argument('--patchlen5', type=int, default=48, help='k seasonals')
parser.add_argument('--patchlen6', type=int, default=30, help='k seasonals')
parser.add_argument('--patchlen7', type=int, default=24, help='k seasonals')
parser.add_argument('--patchlen8', type=int, default=48, help='k seasonals')
parser.add_argument('--weight1', type=float, default=0.5, help='k seasonals')
parser.add_argument('--weight2', type=float, default=0.5, help='k seasonals')
parser.add_argument('--weight3', type=float, default=0.5, help='k seasonals')
parser.add_argument('--weight4', type=float, default=0.365, help='k seasonals')
parser.add_argument('--weight5', type=float, default=0.335, help='k seasonals')
parser.add_argument('--weight6', type=float, default=0.3, help='k seasonals')
parser.add_argument('--weight7', type=float, default=0.365, help='k seasonals')
parser.add_argument('--weight8', type=float, default=0.335, help='k seasonals')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mae', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

#MCRCN
parser.add_argument('--top_k', type=int, default=3, help='k seasonals')
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--d_state', type=int, default=2, help='d_state')
parser.add_argument('--mam_model', type=int, default=24, help='mam_model')
parser.add_argument('--mam_layer', type=int, default=2, help='mam_layer')
parser.add_argument('--num_layer', type=int, default=2, help='num_layer')
parser.add_argument('--kernel_size', type=int, default=4, help='kernel_size')
parser.add_argument('--conv2d_kernel', type=int, default=4, help='kernel_size')
parser.add_argument('--conv2d_kernel2', type=int, default=16, help='kernel_size')
parser.add_argument('--dim', type=int, default=20, help='D')
parser.add_argument('--r', type=int, default=1, help='r')

args = parser.parse_args()
if __name__ == '__main__':
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_gru_model{}_gru_hidden{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.gru_model,
                args.gru_hidden,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_gru_model{}_gru_hidden{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                    args.gru_model,
                                                                                                    args.gru_hidden,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
