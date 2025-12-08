import sys
import mars

import os
import time
import argparse

import numpy as np
import pandas as pd

from mars.utils.config_utils import Config
from mars.utils.file_utils import PathsContainer
from mars.utils.data_utils import cvSplit
from mars.preprocess.gblup import Hiblup
from mars.preprocess.fs import SnpSelect
from mars.models.train import Model, Predict
from sklearn.model_selection import train_test_split


def hyper_parameter():
    ''' Hyper Parameter Setting '''
    parser = argparse.ArgumentParser("MARS")
    parser.add_argument("--json_path", type = str)  # 配置文件路径 
    return parser.parse_args()


def main():
    # 获得超参数
    arg = hyper_parameter()
    start_time = time.time()

    # 创建结果文件夹,读取配置文件
    path = arg.json_path
    paths = PathsContainer.from_args(path)
    config = Config.from_json(os.path.join(path, "config.json"))

    # 读取数据
    features = np.load(config.data.gene) # 基因型文件.npy
    labels = np.array(pd.read_csv(config.data.pheno, sep='\t')) # 有表头的表型文件
    print(features.shape)
    print(labels.shape)
    x_train, y_train, x_test, y_test = cvSplit(features, labels)
    print('Records of training set : {}'.format(x_train.shape[0]))
    print('Records of testing set : {}'.format(x_test.shape[0]))

    # 划分验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=0.2, 
                                                      random_state=config.seed)

    # GBLUP
    y_gebv_train, y_gebv_val, y_gebv_test = Hiblup(labels, y_train, y_val, y_test, 
                                                    config, paths.blup_path)
    print(y_gebv_train.shape)
    print('HIBLUP finished.')

    # 特征选择
    x_train, x_val, x_test = SnpSelect(x_train, y_gebv_train[:,2], x_val, y_gebv_val[:,2],
                                       x_test, config, paths.index_path)
    print('SNP selection finished.')

    # 训练模型
    best_model, model = Model(x_train, y_gebv_train, x_val, y_gebv_val, config, paths)
    print('Model finished.')

    # 微调预测
    x_train = np.concatenate([x_train, x_val], axis=0)
    y_gebv_train = np.concatenate([y_gebv_train, y_gebv_val], axis=0)
    y_pred = Predict(best_model, model, x_train, y_gebv_train, x_test, y_gebv_test, config, paths)
    pred = {
        'ID': y_test[:,0],
        'GEBV': y_gebv_test[:,1],
        'PRED': y_pred.reshape(-1)
    }
    pred = pd.DataFrame(pred)
    pred.to_csv(os.path.join(path, 'prediction.csv'), sep="\t", index=False, header=True, quoting=False)
    print('Prediction finished.')

    end_time = time.time()
    print(f"程序执行时间为: {end_time - start_time} 秒")


if __name__ == '__main__':
    main()
