import os
import torch
import numpy as np
import pandas as pd
import xgboost as xgb

from joblib import dump
from mars.models.dl import dl_train, retrain
from mars.models.ml import svr_train, rf_train, xgb_train, lgb_train, xgb_retrain, lgb_retrain
from mars.utils.file_utils import read_anno
from mars.utils.data_utils import bim_sort, ChannelData


def Model(x_train, y_gebv_train, x_val, y_gebv_val, config, paths):
    '''
    input: y_gebv: np, [y_true, gebv]
    '''

    # SVM Model
    if config.model.SVR is not None:
        svm_model, svm_cor = svr_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths)
    if config.model.SVR is None:
        svm_cor = -1
    
    # RandomForest Model
    if config.model.RF is not None:
        rf_model, rf_cor = rf_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths)
    if config.model.RF is None:
        rf_cor = -1

    # XGBoost Model
    if config.model.XGBoost is not None:
        xgb_model, xgb_cor = xgb_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths)
    if config.model.XGBoost is None:
        xgb_cor = -1

    # LightGBM Model
    if config.model.LightGBM is not None:
        lgb_model, lgb_cor = lgb_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths)
    if config.model.LightGBM is None:
        lgb_cor = -1
    
    # MLP Model
    if config.model.MLP is not None:
        mlp_model, mlp_cor = dl_train(x_train, y_gebv_train, x_val, y_gebv_val, 'MLP', config, paths)
    if config.model.MLP is None:
        mlp_cor = -1
    
    # CNN Model
    if config.model.CNN is not None:
        cnn_model, cnn_cor = dl_train(x_train, y_gebv_train, x_val, y_gebv_val, 'CNN', config, paths)
    if config.model.CNN is None:
        cnn_cor = -1

    # Swim Transformer Model
    if config.model.SwimTransformer is not None:
      swim_model, swim_cor = dl_train(x_train, y_gebv_train, x_val, y_gebv_val, 'SwimTransformer', config, paths)
    if config.model.SwimTransformer is None:
      swim_cor = -1

    result = {
        'SVM': svm_cor,
        'RandomForest': rf_cor,
        'XGBoost': xgb_cor,
        'LightGBM': lgb_cor,
        'MLP': mlp_cor,
        'CNN': cnn_cor,
        'SwimTransformer': swim_cor
    }
    result = pd.DataFrame([result])
    result.to_csv(os.path.join(paths.pred_path, 'comparison_val.csv'), sep=",", index=False, header=True, quoting=False)
    best_model = result.iloc[0].idxmax()
    print('The best model: %s' % best_model)

    # 根据best_model的名称输出相应的模型实例
    if best_model == 'SVM':
        return best_model, svm_model
    elif best_model == 'RandomForest':
        return best_model, rf_model
    elif best_model == 'XGBoost':
        return best_model, xgb_model
    elif best_model == 'LightGBM':
        return best_model, lgb_model
    elif best_model == 'MLP':
        return best_model, mlp_model
    elif best_model == 'CNN':
        return best_model, cnn_model
    elif best_model == 'SwimTransformer':
        return best_model, swim_model


def Predict(best_model, model, x_train, y_gebv_train, x_test, y_gebv_test, config, paths):
    if best_model == 'SVM' or best_model == 'RandomForest':
        model.fit(x_train, y_gebv_train[:,2].astype(float))
        dump(model, os.path.join(paths.weight_path, 'rf.joblib'))
        y_pred = model.predict(x_test).reshape(-1)
        y_pred += y_gebv_test[:,1].astype(float)

    elif best_model == 'XGBoost':
        model = xgb_retrain(x_train, y_gebv_train[:,2], model, config, paths)
        dtest = xgb.DMatrix(x_test)
        y_pred = model.predict(dtest).reshape(-1)
        y_pred += y_gebv_test[:,1].astype(float)

    elif best_model == 'LightGBM':
        model = lgb_retrain(x_train, y_gebv_train[:,2], model, config, paths)
        y_pred = model.predict(x_test).reshape(-1)
        y_pred += y_gebv_test[:,1].astype(float)

    elif best_model == 'MLP' or best_model == 'CNN' or best_model == 'SwimTransformer':
        if best_model == 'SwimTransformer':
            if config.data.anno is not None:
                anno = read_anno(config.data.anno)
            else: # 如果没有注释信息
                bim = pd.read_csv(f'{config.data.plink}.bim', header=None, sep='\t')
                anno = bim_sort(bim)
            x_train = ChannelData(x_train, anno, paths.index_path)
            x_test = ChannelData(x_test, anno, paths.index_path)
        
        configs = getattr(config.model, best_model)
        model = retrain(x_train, y_gebv_train, model, best_model, configs, paths)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_pred = model(x).cpu().detach()
        y_pred = np.array(y_pred).reshape(-1) + y_gebv_test[:,1]
    return y_pred
