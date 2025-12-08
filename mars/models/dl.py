import os
import math
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR

from mars.utils.file_utils import read_anno
from mars.utils.data_utils import myDataset, picture, set_seed, bim_sort, ChannelData, re_picture
from mars.models.one_dimensional_swin import CirculateSwinBlock


def dl_train(x_train, y_gebv_train, x_val, y_gebv_val, model_name, config, paths):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using {} device.'.format(device))

    set_seed(config.seed)
    configs = getattr(config.model, model_name)
    batch_size = configs['batch_size']
    lr = configs['lr']
    wd = configs['weight_decay']
    num_epochs = configs['epochs']

    # bulid model实例化网络
    if model_name == 'MLP':
        net = MLP(x_train.shape[-1], configs).to(device)
    elif model_name == 'CNN':
        net = CNN(x_train.shape[-1], configs).to(device)
    elif model_name == 'SwimTransformer':
        if config.data.anno is not None:
            anno = read_anno(config.data.anno)
        else: # 如果没有注释信息
            bim = pd.read_csv(f'{config.data.plink}.bim', header=None, sep='\t')
            anno = bim_sort(bim)
        x_train = ChannelData(x_train, anno, paths.index_path)
        x_val = ChannelData(x_val, anno, paths.index_path)
        net = Swim_Transformer(x_train.shape, configs).to(device)
    else:  
        raise ValueError("Unsupported model. Please use 'MLP', 'CNN' or 'Swim Transformer'.") 

    if configs['loss'] == 'mse':  
        loss = nn.MSELoss()  
    elif configs['loss'] == 'mae':  
        loss = nn.L1Loss()
    else:  
        raise ValueError("Unsupported loss function. Please use 'mse' or 'mae'.") 

    if configs['optimizer'] == 'adam':
        trainer = torch.optim.Adam(net.parameters(), lr=lr) # adam没有wd
    elif configs['optimizer'] == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError("Unsupported loss function. Please use 'adam' or 'sgd'.") 
    
    scheduler = CosineAnnealingLR(trainer, T_max=50, eta_min=lr*0.1)

    print(x_train.shape)
    train_set = myDataset(x_train, y_gebv_train)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=0, #config.GBLUP.threads
                              shuffle=True, drop_last=False)

    val_set = myDataset(x_val, y_gebv_val)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=0, #config.GBLUP.threads
                            shuffle=False, drop_last=False)

    train_epoch_loss = []
    train_epoch_cor = []
    train_epoch_mse = []
    val_epoch_loss = []
    val_epoch_cor = []
    val_epoch_mse = []
    max_val_cor = -1
    maxcor_epoch = num_epochs
    for epoch in range(num_epochs):
        # train
        net.train()
        loss_train = 0
        y_pred_step = []
        y_step = []
        step_loss = []

        for step, (x,y,g,r) in enumerate(train_loader):
            x = x.to(device)
            r = r.reshape(-1,1).to(device)
            trainer.zero_grad() #梯度清空
            y_pred = net(x).reshape(-1,1)
            l_train = loss(y_pred, r)
            l_train.backward()
            trainer.step() #更新模型
            scheduler.step()
            loss_train += l_train.item()
            step_loss.append(l_train.item())
            y_pred = y_pred.cpu().detach().reshape(-1) + g
            y_pred_step.append(y_pred)
            y_step.append(y.reshape(-1))

        loss_train /= step + 1
        train_epoch_loss.append(loss_train)
        y_pred = torch.cat(y_pred_step, dim=0).numpy().reshape(-1)
        y = torch.cat(y_step, dim=0).numpy().reshape(-1)
        cor_train, _ = pearsonr(y_pred, y)
        train_epoch_cor.append(cor_train)
        mse_train = mean_squared_error(y_pred, y)
        train_epoch_mse.append(mse_train)

        # val
        net.eval()
        loss_val = 0
        y_pred_step = []
        y_step = []
        with torch.no_grad():
            for step, (x,y,g,r) in enumerate(val_loader):
                x = x.to(device)
                r = r.reshape(-1,1).to(device)
                y_pred = net(x).reshape(-1,1)
                loss_val += loss(y_pred, r).item()
                y_pred = y_pred.cpu().detach().reshape(-1) + g
                y_pred_step.append(y_pred)
                y_step.append(y.reshape(-1))

            loss_val /= step + 1
            val_epoch_loss.append(loss_val)
            val_pred = torch.cat(y_pred_step, dim=0).numpy().reshape(-1)
            y = torch.cat(y_step, dim=0).numpy().reshape(-1)
            cor_val, _ = pearsonr(val_pred, y)
            val_epoch_cor.append(cor_val)
            mse_val = mean_squared_error(val_pred, y)
            val_epoch_mse.append(mse_val)

        print(f'epoch {epoch + 1}, loss {loss_train:f}, cor {cor_train:f}; val loss {loss_val:f}, val cor {cor_val:f}.')
        if (epoch + 1) % 1 == 0: #50
            picture(train_epoch_cor, val_epoch_cor, train_epoch_mse, val_epoch_mse,
                    train_epoch_loss, val_epoch_loss,
                    model_name, paths.plot_path)

        if cor_val >= max_val_cor:
            max_val_cor = cor_val
            maxcor_mse = mse_val
            maxcor_train_cor = cor_train
            maxcor_train_mse = mse_train
            weight = net.state_dict()
            maxcor_epoch = epoch + 1
            torch.save(weight, os.path.join(paths.weight_path, f"{model_name}_best_cor.pth"))
            maxcor_val_pred = val_pred
            pd.DataFrame(maxcor_val_pred).to_csv(os.path.join(paths.pred_path, f"{model_name}_val_pred.txt"), index=False, header=False)
            plt.figure(figsize=(10, 7))
            plt.hist(maxcor_val_pred, bins=30, edgecolor='black')
            plt.savefig(os.path.join(paths.pred_path, f"{model_name}_val_pred.png"))

        if epoch > maxcor_epoch + configs['early_stopping'] :
            picture(train_epoch_cor, val_epoch_cor, train_epoch_mse, val_epoch_mse,
                    train_epoch_loss, val_epoch_loss,
                    model_name, paths.plot_path)
            print(f"Early stopping triggered at epoch {epoch}. Best model was at epoch {maxcor_epoch}")
            break

    result = np.array([int(maxcor_epoch), maxcor_train_cor, maxcor_train_mse, max_val_cor, maxcor_mse]).reshape(1,-1)
    result = pd.DataFrame(result, columns=['best_epoch', 'train_cor', 'train_mse', 'val_cor', 'val_mse'])
    result.to_csv(os.path.join(paths.pred_path, f'{model_name}_result.csv'), index=False, sep=',')

    return net, max_val_cor


def retrain(x_train, y_gebv_train, net, net_name, configs, paths):
    # 冻结预训练模型的卷积层
    weight = os.path.join(paths.weight_path, f"{net_name}_best_cor.pth")
    net.load_state_dict(torch.load(weight), strict=False)
    for param in net.parameters():
        param.requires_grad = False
    for param in net.output.parameters():
        param.requires_grad = True

    lr = configs['lr'] * 0.01
    wd = configs['weight_decay']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if configs['loss'] == 'mse':  
        loss = nn.MSELoss()  
    elif configs['loss'] == 'mae':  
        loss = nn.L1Loss()
    else:  
        raise ValueError("Unsupported loss function. Please use 'mse' or 'mae'.") 

    if configs['optimizer'] == 'adam':
        trainer = torch.optim.Adam(net.parameters(), lr=lr) # adam没有wd
    elif configs['optimizer'] == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError("Unsupported loss function. Please use 'adam' or 'sgd'.") 
    
    train_set = myDataset(x_train, y_gebv_train)
    train_loader = DataLoader(dataset=train_set, batch_size=128, num_workers=0, #config.GBLUP.threads
                              shuffle=True, drop_last=False)

    train_epoch_loss = []
    train_epoch_cor = []
    train_epoch_mse = []
    max_cor = -1
    maxcor_epoch = 100
    for epoch in range(100):
        # train
        net.train()
        loss_train = 0
        y_pred_step = []
        y_step = []
        step_loss = []

        for step, (x,y,g,r) in enumerate(train_loader):
            x = x.to(device)
            r = r.reshape(-1,1).to(device)
            trainer.zero_grad() #梯度清空
            y_pred = net(x).reshape(-1,1)
            l_train = loss(y_pred, r)
            l_train.backward()
            trainer.step() #更新模型
            loss_train += l_train.item()
            step_loss.append(l_train.item())
            y_pred = y_pred.cpu().detach().reshape(-1) + g
            y_pred_step.append(y_pred)
            y_step.append(y.reshape(-1))

        loss_train /= step + 1
        train_epoch_loss.append(loss_train)
        y_pred = torch.cat(y_pred_step, dim=0).numpy().reshape(-1)
        y = torch.cat(y_step, dim=0).numpy().reshape(-1)
        cor_train, _ = pearsonr(y_pred, y)
        train_epoch_cor.append(cor_train)
        mse_train = mean_squared_error(y_pred, y)
        train_epoch_mse.append(mse_train)

        print(f'epoch {epoch + 1}, loss {loss_train:f}, cor {cor_train:f}.')
        if (epoch + 1) % 1 == 0: #50
            re_picture(train_epoch_cor, train_epoch_loss, net_name, paths.plot_path)
        
        if cor_train >= max_cor:
            max_cor = cor_train
            weight = net.state_dict()
            maxcor_epoch = epoch + 1
            torch.save(weight, os.path.join(paths.weight_path, f"best_retrain_{net_name}.pth"))

        if epoch > maxcor_epoch + 10:
            re_picture(train_epoch_cor, train_epoch_loss, net_name, paths.plot_path)
            print(f"Early stopping triggered at epoch {epoch}. Best model was at epoch {maxcor_epoch}")
            break
    return net


class MLP(nn.Module):
    def __init__(self, nsnp, config):
        super(MLP, self).__init__()
        self.input = nn.Sequential(nn.Linear(nsnp, 2048),
                                   nn.BatchNorm1d(2048),
                                   nn.Dropout(config['dropout']))
        
        layers = config['layers']
        hidden = []
        for i in range(layers):
            out_size = 2048 // (4**(i+1))
            hidden.append(nn.Sequential(nn.Linear(out_size*4, out_size),
                                        nn.BatchNorm1d(out_size)))
        self.hidden = nn.Sequential(*hidden)
        
        size = 2048 // (4**layers)
        self.output = nn.Sequential(nn.ReLU(),
                                    nn.Linear(size, 1))

    def forward(self, x):
        y = self.input(x)
        y = self.hidden(y)
        y = self.output(y)
        return y


class CNN(nn.Module):
    def __init__(self, nsnp, config):
        super(CNN, self).__init__()

        def calc_output_size(input_size, kernel_size, stride, channel, pool_size=1):
            out1 = ((input_size - kernel_size) // stride + 1 ) // pool_size
            out2 = ((out1 - kernel_size) // stride + 1 ) // 1
            out3 = ((out2 - kernel_size) // stride + 1 ) // pool_size
            return out3 * channel
        
        # 动态计算线性层输入大小
        size = calc_output_size(nsnp, 6, 2, 8, 3)
        self.simple = nn.Sequential(nn.Conv1d(1, 8, kernel_size=6, stride=2),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=3),
                                    nn.BatchNorm1d(8),
                                    nn.Conv1d(8, 8, kernel_size=6, stride=2),
                                    nn.Conv1d(8, 8, kernel_size=6, stride=2),
                                    nn.AvgPool1d(kernel_size=3),
                                    nn.Flatten(),
                                    nn.Dropout(config['dropout']),
                                    nn.Linear(size, 128),
                                    nn.ReLU())
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.simple(x)
        y = self.output(y)
        return y


class Swim_Transformer(nn.Module):
    def __init__(self, shape, config):
        super(Swim_Transformer, self).__init__()

        dim = shape[2]  #特征数
        window_size = config['window_size']
        hidden = math.ceil(shape[1] / window_size)
        stages = [
            (
                dim, #lay_num
                True, #squeeze
                window_size,
            ),
            (
                2,
                False,
                window_size,
            ),
            (
                2,
                False,
                window_size,
            ),
            (
                2,
                False,
                window_size,
            ),
        ]
        self.net = CirculateSwinBlock(stages, dim)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        y = self.net(x)
        y = self.output(y)
        return y
