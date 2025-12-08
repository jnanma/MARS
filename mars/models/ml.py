import os
import xgboost as xgb
import lightgbm as lgb

from joblib import dump
from sklearn.svm import SVR
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

from mars.utils.data_utils import set_seed


def svr_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths):
    set_seed(config.seed)
    model = SVR(kernel=config.model.SVR['kernel'])
    model.fit(x_train, y_gebv_train[:,2].astype(float))
    dump(model, os.path.join(paths.weight_path, 'svr.joblib'))
    y_pre = model.predict(x_val).reshape(-1)
    y_pre += y_gebv_val[:,1].astype(float)
    r, _ = pearsonr(y_gebv_val[:,0], y_pre)
    return model, r


def rf_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths):
    set_seed(config.seed)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(x_train, y_gebv_train[:,2].astype(float))
    dump(model, os.path.join(paths.weight_path, 'rf.joblib'))
    y_pre = model.predict(x_val).reshape(-1)
    y_pre += y_gebv_val[:,1].astype(float)
    r, _ = pearsonr(y_gebv_val[:,0], y_pre)
    return model, r


def xgb_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths):

    dtrain = xgb.DMatrix(x_train, y_gebv_train[:,2].astype(float))
    dval = xgb.DMatrix(x_val, y_gebv_val[:,2].astype(float))

    # 初始参数
    set_seed(config.seed)
    configs = config.model.XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        "max_depth": configs['max_depth'],
        "eta": configs['eta'],
        "lambda": configs['lm'],
        'n_jobs': -1
    }
    model = xgb.train(params=params, dtrain=dtrain, 
                      evals=[(dtrain,'train'),(dval,'valid')],
                      num_boost_round=configs['xgbround'], 
                      early_stopping_rounds=configs['early_stopping'])
    model.save_model(os.path.join(paths.weight_path, 'xgb.json'))
    y_pre = model.predict(dval).reshape(-1)
    y_pre += y_gebv_val[:,1].astype(float)
    r, _ = pearsonr(y_gebv_val[:,0], y_pre)
    return model, r


def lgb_train(x_train, y_gebv_train, x_val, y_gebv_val, config, paths):

    dtrain = lgb.Dataset(x_train, y_gebv_train[:,2].astype(float))
    dval = lgb.Dataset(x_val, y_gebv_val[:,2].astype(float))

    # 初始参数
    set_seed(config.seed)
    configs = config.model.LightGBM
    params = {
        'objective': 'regression',
        'eval_metric': 'l2',
        'learning_rate': configs['learning_rate'],
        'lambda_l2': configs['lambda_l2'],
        'num_threads': -1,
    }
    callbacks = [lgb.log_evaluation(period=1), 
                 lgb.early_stopping(stopping_rounds=configs['early_stopping'])]

    model = lgb.train(params, dtrain, valid_sets=dval, 
                      num_boost_round=configs['round'],
                      callbacks=callbacks) #测试完修改early_stopping
    model.save_model(os.path.join(paths.weight_path, 'lgb.txt'))
    y_pre = model.predict(x_val).reshape(-1)
    y_pre += y_gebv_val[:,1].astype(float)
    r, _ = pearsonr(y_gebv_val[:,0], y_pre)
    return model, r


def xgb_retrain(x_train, r_train, model, config, paths):
    dtrain = xgb.DMatrix(x_train, r_train.astype(float))
    # 初始参数
    set_seed(config.seed)
    configs = config.model.XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        "max_depth": configs['max_depth'],
        "eta": configs['eta'] * 0.01,
        "lambda": configs['lm'],
        'n_jobs': -1
    }
    model = xgb.train(params=params, dtrain=dtrain, 
                      num_boost_round=100, evals=[(dtrain, 'eval')],
                      early_stopping_rounds=10, xgb_model=model)
    model.save_model(os.path.join(paths.weight_path, 'best_retrain_xgb.jsosn'))
    return model


def lgb_retrain(x_train, r_train, model, config, paths):
    dtrain = lgb.Dataset(x_train, r_train.astype(float))
    set_seed(config.seed)
    configs = config.model.LightGBM
    params = {
        'objective': 'regression',
        'eval_metric': 'l2',
        'learning_rate': configs['learning_rate'] * 0.01,
        'lambda_l2': configs['lambda_l2'],
        'num_threads': -1,
    }
    callbacks = [lgb.log_evaluation(period=1), 
                 lgb.early_stopping(stopping_rounds=10)]
    model = lgb.train(params=params, train_set=dtrain, 
                      num_boost_round=100, valid_sets=dtrain,
                      callbacks=callbacks, init_model=model)
    model.save_model(os.path.join(paths.weight_path, 'best_retrain_lgb.json'))
    return model
