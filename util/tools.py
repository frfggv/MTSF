import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
def visual1(seasonal, trend=None, true=None,pred=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()

    plt.plot(seasonal, label='Seasonal', linewidth=2, color='red')
    plt.plot(trend, label='Trend', linewidth=2, color='blue')
    plt.plot(true, label='GroundTruth', linewidth=2, color='grey')
    plt.plot(pred, label='Prediction', linewidth=2, color='goldenrod')
    plt.grid(True)

    plt.legend()
    plt.savefig(name, bbox_inches='tight')
def visual11(seasonal, trend=None, true=None,pred=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()

    plt.plot(seasonal, label='Seasonal', linewidth=2, color='red')
    plt.plot(trend, label='Trend', linewidth=2, color='blue')
    plt.plot(pred, label='Prediction', linewidth=2, color='goldenrod')
    plt.grid(True)

    plt.legend()
    plt.savefig(name, bbox_inches='tight')
def visual_truth_predict(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color='goldenrod')
    plt.plot(true, label='GroundTruth', linewidth=2, color='grey')
    plt.grid(True)
    # 在横坐标为 96 的地方画一条竖线
    # plt.axvline(x=720, linestyle='—— ——', color='red', linewidth=0.7)  # 可以更改 linestyle 和 color 参数
    plt.legend()
    # plt.savefig(name, dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(name, bbox_inches='tight')
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color='goldenrod')
    plt.plot(true, label='GroundTruth', linewidth=2, color='grey')
    true_lookback=true[:96]
    plt.plot(true_lookback, label='LookBack', linewidth=2, color='black')
    plt.grid(True)
    # 在横坐标为 96 的地方画一条竖线
    # plt.axvline(x=720, linestyle='—— ——', color='red', linewidth=0.7)  # 可以更改 linestyle 和 color 参数
    plt.legend()
    # plt.savefig(name, dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(name, bbox_inches='tight')
def visual1_ablation(seasonal, trend=None, true=None,pred=None, name='./pic/test_abla.pdf'):
    """
    Results visualization
    """
    plt.figure()

    plt.plot(seasonal, label='Seasonal', linewidth=2, color='red')
    plt.plot(trend, label='Trend', linewidth=2, color='blue')
    plt.plot(true, label='GroundTruth', linewidth=2, color='grey')
    plt.grid(True)

    plt.legend()
    plt.savefig(name, bbox_inches='tight')
def visual_ablation(true, preds=None, name='./pic/test_abla.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, color='goldenrod')
    plt.plot(true, label='GroundTruth', linewidth=2, color='grey')


    plt.grid(True)
    # 在横坐标为 96 的地方画一条竖线
    # plt.axvline(x=720, linestyle='—— ——', color='red', linewidth=0.7)  # 可以更改 linestyle 和 color 参数
    plt.legend()
    # plt.savefig(name, dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(name, bbox_inches='tight')
# def visual(true, preds=None, name='./pic/test'):
#     """
#     Results visualization
#     """
#     plt.figure()
#     plt.plot(true, label='GroundTruth', linewidth=2,  color='grey')
#     if preds is not None:
#         plt.plot(preds, label='Prediction', linewidth=2,  color='goldenrod')
#     # plt.plot(true, label='GroundTruth', linewidth=2)
#     # plt.axvline(x=720, linestyle='—— ——', color='grey', linewidth=0.6)  # 可以更改 linestyle 和 color 参数
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(name,  dpi=600, format='svg', bbox_inches='tight')
#     # plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))