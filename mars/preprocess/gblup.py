import os
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def Hiblup(labels, y_train, y_val, y_test, config, path):
    '''
    output: np.array, [y_true, GA, residuals]
    '''
    trait = config.trait
    # 将validation set的表型值NA
    index = np.where(np.isin(labels[:, 0], y_test[:, 0]))[0]
    labels[index, 1] = 'NA'
    index = np.where(np.isin(labels[:, 0], y_test[:, 0]))[0]
    labels[index, 1] = 'NA'
    labels_new = pd.DataFrame(labels, columns=['id', str(trait)])
    labels_new_path = os.path.join(path, f'{trait}_val.phe')
    labels_new.to_csv(labels_new_path, index=False, header=True, quoting=False, sep='\t')

    # 运行Hiblup脚本
    command = [
        config.GBLUP.hiblup,
        "--single-trait",
        "--threads", str(config.GBLUP.threads),
        "--pheno", labels_new_path,
        "--pheno-pos", "2",
        "--bfile", config.data.plink, 
        "--out", os.path.join(path, trait)
    ]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("标准输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("命令执行失败，错误码:", e.returncode)
        print("错误输出:", e.stderr)

    miu = pd.read_table(os.path.join(path, f"{trait}.beta"), sep='\t')
    miu = miu.iloc[0, 1]  # 获取第一行第二列的miu值

    gebv = pd.read_table(os.path.join(path, f"{trait}.rand"), sep='\t')
    gebv.iloc[:, 1] += miu

    # 匹配个体，读取gebv
    gebv_train = merge_gebv(y_train, gebv)
    gebv_val = merge_gebv(y_val, gebv)
    gebv_test = merge_gebv(y_test, gebv)

    # 计算相关系数和均方误差
    cor_mse(gebv_train, gebv_val, path)

    return gebv_train, gebv_val, gebv_test


def merge_gebv(y, gebv):
    '''
    y: np.array, [id, y_true]
    gebv: pd.dataframe, [ID, GA, residuals]
    output: np.array, [y_true, GA, residuals]
    '''
    y = pd.DataFrame(y, columns=['id', 'y'])
    merged = pd.merge(y, gebv, left_on='id', right_on='ID', how='left')
    output = np.array(merged[['y', 'GA']]).astype(float)
    resi = (output[:,0] - output[:,1]).reshape(-1,1)
    output = np.concatenate([output,resi], axis=1)
    print(output.shape)
    return output


def cor_mse(gebv_train, gebv_val, path):
    train_cor, _ = pearsonr(gebv_train[:,0], gebv_train[:,1])
    train_mse = mean_squared_error(gebv_train[:,0], gebv_train[:,1])
    val_cor, _ = pearsonr(gebv_val[:,0], gebv_val[:,1])
    val_mse = mean_squared_error(gebv_val[:,0], gebv_val[:,1])
    result = {
        'train_cor': train_cor,
        'train_mse': train_mse,
        'val_cor': val_cor,
        'val_mse': val_mse
    }
    result_df = pd.DataFrame([result])
    result_df.to_csv(os.path.join(path, 'gblup.csv'), sep="\t", index=False, header=True, quoting=False)
