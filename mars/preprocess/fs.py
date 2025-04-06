import re
import os
import random

import numpy as np
import pandas as pd
import xgboost as xgb

from mars.utils.data_utils import bim_sort
from mars.utils.file_utils import read_anno


def SnpSelect(x_train, y_train, x_val, y_val, x_test, 
              config, path):
    bim = pd.read_csv(f'{config.data.plink}.bim', header=None, sep='\t') # plink文件bim
    if config.data.anno is not None:
        anno = read_anno(config.data.anno)
    else: # 如果没有注释信息
        anno = bim_sort(bim)

    index_id_list = []
    for t in range(len(anno)):
        id = anno[t].reshape(-1)
        if len(id) <= 100:
            index_id = id
        elif len(id) > 100:
            fs_train, id_sorted = id_search(x_train, bim, id)
            fs_val, _ = id_search(x_val, bim, id)
            print()
            index_id = select(fs_train, y_train, fs_val, y_val, id_sorted, config, path, t)
        print(f'Records of index {t+1} : {len(index_id)}.')
        index_id_list.append(index_id)
    index_id_list = np.concatenate(index_id_list, axis=0)

    # 按照bim的顺序保存
    index = bim.iloc[:, 1].isin(index_id_list)
    index_id_list = bim.loc[index, 1].to_numpy()
    pd.DataFrame(index_id_list).to_csv(os.path.join(path, 'index.snplist'), index=False, header=None, sep='\t')

    x_train_sub, _ = id_search(x_train, bim, index_id_list)
    x_val_sub, _ = id_search(x_val, bim, index_id_list)
    x_test_sub, _ = id_search(x_test, bim, index_id_list)
    
    return x_train_sub, x_val_sub, x_test_sub


def select(x_train, y_train, x_val, y_val, id, config, path, t):

    dtrain = xgb.DMatrix(x_train, y_train)
    dval = xgb.DMatrix(x_val, y_val)

    # 初始参数
    random.seed(config.seed)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": config.SnpSelect.max_depth,
        "eta": config.SnpSelect.eta,
        "subsample": config.SnpSelect.subsample,
        "lambda": config.SnpSelect.lm,
        'n_jobs': 8
    }

    model = xgb.train(params=params, dtrain=dtrain, 
                      evals=[(dtrain,'train'),(dval,'valid')],
                      num_boost_round=config.SnpSelect.xgbround, 
                      early_stopping_rounds=config.SnpSelect.early_stopping, 
                      custom_metric=correlation_coefficient, maximize=True)
    
    model.save_model(os.path.join(path, f'SnpSelect{t+1}_model.json'))

    # 获取特征的 gain 值
    gains = model.get_score(importance_type='gain')
    index = [int(re.sub(r"\D", "", index)) for index, gain in gains.items() if gain > 0]
    index_id = id.iloc[index]

    return index_id


def correlation_coefficient(preds, dtrain):
    labels = dtrain.get_label()
    correlation = np.corrcoef(labels, preds)[0, 1]
    return 'correlation', correlation


def id_search(data, bim, annoid):
    index = bim.iloc[:, 1].isin(annoid)
    annoid = bim.loc[index, 1]
    data_sub = data[:, index]
    return data_sub, annoid
