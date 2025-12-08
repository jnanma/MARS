import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from mars.utils.file_utils import read_anno


def set_seed(seed):
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 需要确定性（即每次运行结果相同），需要关闭基准测试
    torch.backends.cudnn.deterministic = True


def cvSplit(features, labels):
    '''
    output:
    test_labels/train_labels : [id, value], 'object'
    test_fea/train_fea : [snps]
    '''
    y = labels[:, 1].astype(float)
    index = np.where(np.isnan(y))[0]
    test_labels = labels[index,:]
    test_fea = features[index,:]

    not_index = np.where(~np.isnan(y))[0] 
    train_labels = labels[not_index,:]
    train_fea = features[not_index,:]

    return train_fea, train_labels, test_fea, test_labels


def bim_sort(bim):
    chr = bim.iloc[:,0].unique()
    snplist = []
    for i in chr:
        indices = bim.index[bim.iloc[:, 0] == i].tolist()
        list = np.array(bim.iloc[indices,1])
        snplist.append(list)
    return snplist


def ChannelData(data, anno, index_path):
    '''
    data: torch (sample, snps)
    anno: anno/bim
    '''
    fid = pd.read_csv(os.path.join(index_path, 'index.snplist'), header=None, sep='\t') # 按bim顺序保存的snpid
    xx = []
    for t in range(len(anno)):
        x = np.zeros_like(data)  # 初始化 x1
        id = anno[t].reshape(-1) 
        index = fid.iloc[:,0].isin(id).values
        x[:, index] = 1
        xx.append(x)
    data = np.expand_dims(data, axis=2)
    xx = np.stack(xx, axis=2)
    xx = np.concatenate((data, xx), axis=2)
    return xx


def picture(train_epoch_cor, test_epoch_cor,
            train_epoch_mse, test_epoch_mse,
            train_epoch_loss, test_epoch_loss,
            model_name, path):
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    plt.title('PCC')
    plt.plot(train_epoch_cor, color='green', label='train cor')
    plt.plot(test_epoch_cor, color='#87CEEB', label='test cor')
    plt.xlabel('Epochs')
    plt.ylabel('Cor')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    plt.title('MSE')
    plt.plot(train_epoch_mse, color='green', label='train mse')
    plt.plot(test_epoch_mse, color='#87CEEB', label='test mse')
    plt.xlabel('Epochs')
    plt.ylabel('Mse')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.savefig(os.path.join(path, f"{model_name}_Epochs-CorMse.png"))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_epoch_loss, color='orange', label='train loss')
    plt.plot(test_epoch_loss, color='red', label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"{model_name}_Epochs-Loss.png"))


def re_picture(train_epoch_cor, train_epoch_loss, model_name, path):
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    plt.title('PCC')
    plt.plot(train_epoch_cor, color='green', label='train cor')
    plt.xlabel('Epochs')
    plt.ylabel('Cor')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    plt.title('Loss')
    plt.plot(train_epoch_loss, color='orange', label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.savefig(os.path.join(path, f"re_{model_name}_Epochs-CorLoss.png"))


class myDataset(Dataset):
    def __init__(self, features, y_gebv):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(y_gebv[:, 0].astype(float), dtype=torch.float32)
        self.gebvs = torch.tensor(y_gebv[:, 1].astype(float), dtype=torch.float32)
        self.resi = torch.tensor(y_gebv[:, 2].astype(float), dtype=torch.float32)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.gebvs[index], self.resi[index]

    def __len__(self):
        return len(self.labels)
