from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, DLinear, MCRCN, PatchTST, TCN

from util.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop,visual1,visual_ablation,visual1_ablation,visual11,visual_truth_predict
from util.metrics import metric
from layers.PatchTST_layers import series_decomp
import math
import numpy as np
from layers.PatchTST_layers import series_decomp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
def kl_divergence_loss(P, Q):
    """
    计算两个形状为 (b, t, c) 的张量之间的 KL 散度损失

    参数:
    P: 概率分布张量，形状为 (b, t, c)
    Q: 概率分布张量，形状为 (b, t, c)

    返回:
    KL散度损失值
    """
    # 确保 P 和 Q 是有效的概率分布
    P = F.softmax(P, dim=-1)
    Q = F.softmax(Q, dim=-1)

    # 计算 Q 的对数概率分布
    log_Q = torch.log(Q + 1e-10)  # 加上一个小的常数，避免log(0)的情况

    # 计算 KL 散度，reduction='batchmean' 表示对整个批次的平均
    kl_div = F.kl_div(log_Q, P, reduction='batchmean')

    return kl_div
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y, y_pred):
        loss = torch.pow(abs(y - y_pred), 2)
        average_loss = loss.sum()

        # loss = torch.pow(torch.abs(y - y_pred), 2)
        # loss_mean = loss.sum()
        # normalized_loss = loss_mean / 10000  # 归一化处理
        # average_loss = torch.exp(normalized_loss)

        return average_loss

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.decomp_module = series_decomp(args.kernel_size)
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'TCN':TCN,
            'MCRCN':MCRCN
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        return criterion
    def _select_criterion_MSE(self):
        criterion = nn.MSELoss()
        return criterion
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 1:
                        trend_mamba1, trend_mamba2, restored_trend_mamba3, res_mamba1, res_mamba2, restored_res_mamba3, p_ij_trend_mam1,\
                        p_ij_trend_mam3, p_ij_res_mam1,p_ij_res_mam3,outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # outputs,seasonal,trend = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,seasonal,trend = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        if self.args.loss=='mae':
            criterion = self._select_criterion()
        else:criterion = self._select_criterion_MSE()
        criterion_mse=self._select_criterion_MSE()
        # criterion = CustomLoss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            total_tain=0.0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 1:
                        start = time.time()
                        # outputs,seasonal,trend = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        trend_mamba1, trend_mamba2, restored_trend_mamba3, res_mamba1, res_mamba2, restored_res_mamba3, p_ij_trend_mam1, \
                        p_ij_trend_mam3, p_ij_res_mam1, p_ij_res_mam3, outputs = self.model(batch_x, batch_x_mark,
                                                                                            dec_inp, batch_y_mark)
                        total_tain += time.time() - start


                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    loss_recon_trend=criterion_mse(trend_mamba1,trend_mamba2)
                    loss_recon_res = criterion_mse(res_mamba1, res_mamba2)
                    loss_kl_trend = kl_divergence_loss(p_ij_trend_mam1, p_ij_trend_mam3)
                    loss_kl_res = kl_divergence_loss(p_ij_res_mam1, p_ij_res_mam3)
                    # loss_true=loss_recon_trend+loss_recon_res+loss_kl_trend+loss_kl_res+loss
                    loss_true = loss+loss_recon_trend+loss_recon_res+loss_kl_trend+loss_kl_res

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss_true.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("Training Time:{}".format(total_tain))
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.decomp_module = series_decomp(12)
        self.decomp_module2 = series_decomp(24)
        self.decomp_module3 = series_decomp(48)
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        folder_path1 = './test_results/' + 'trend_seasonal_before_normed'+setting + '/'
        if not os.path.exists(folder_path1):
            os.makedirs(folder_path1)
        folder_path2 = './test_results/' + 'Normed_predict_truth' + setting + '/'
        if not os.path.exists(folder_path2):
            os.makedirs(folder_path2)

        total=0.0
        iter=0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 1:
                            start=time.time()
                            # outputs,seasonal,trend = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            trend_mamba1, trend_mamba2, restored_trend_mamba3, res_mamba1, res_mamba2, restored_res_mamba3, p_ij_trend_mam1, \
                            p_ij_trend_mam3, p_ij_res_mam1, p_ij_res_mam3, outputs = self.model(batch_x, batch_x_mark,
                                                                                                dec_inp, batch_y_mark)
                            total+=time.time()-start
                            # iter=i+1
                    # else:
                    #     if self.args.output_attention:
                    #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    #
                    #     else:
                    #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)

                # outputs_seasonal = seasonal[:, -self.args.pred_len:, f_dim:]
                # outputs=outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # res_init, trend_init1 = self.decomp_module(batch_y)
                # res_init2, trend_init2 = self.decomp_module2(batch_y)
                # res_init3, trend_init3 = self.decomp_module3(batch_y)
                # print('res1:',res_init)
                # print('trend1:', trend_init1)
                # print('res2:', res_init2)
                # print('trend2:', trend_init2)
                # print('res3:', res_init3)
                # print('trend3:', trend_init3)
                # seasonal_out, trend_out = self.decomp_module(batch_y)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # seasonal= seasonal.detach().cpu().numpy()
                # trend=trend.detach().cpu().numpy()
                # outputs_seasonal = outputs_seasonal.detach().cpu().numpy()
                # seasonal_out = seasonal_out.detach().cpu().numpy()
                # trend_out = trend_out.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()



                preds.append(pred)
                trues.append(true)
                # if i==0:
                #     print(true[0, :, -1])
                #     print(pred[0, :, -1])
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    #
                    gt = true[0,:,-1]
                    pd = pred[0,:,-1]
                    # folder_path='./test_results/' + 'abla_Attention'+setting + '/'
                    # # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # visual1_ablation(seasonal[0, :, -1], trend[0, :, -1], true[0, :, -1], pred[0, :, -1],
                    #                  os.path.join(folder_path, str(i) + '.pdf'))


                    folder_path = folder_path1
                    # visual11(seasonal[0, :, -1], trend[0, :, -1], true[0, :, -1], seasonal[0, :, -1]+trend[0, :, -1],
                    #         os.path.join(folder_path, str(i) + '.pdf'))
                    folder_path = folder_path2
                    visual_truth_predict(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        infer_time=total
        print('\tinfer_time: {:.4f}s'.format(infer_time))
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs,seasonal,trend = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            trend_mamba1, trend_mamba2, restored_trend_mamba3, res_mamba1, res_mamba2, restored_res_mamba3, p_ij_trend_mam1, \
                            p_ij_trend_mam3, p_ij_res_mam1, p_ij_res_mam3, outputs = self.model(batch_x, batch_x_mark,
                                                                                                dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
