import os
import numpy as np
import pandas as pd
from attr import attrib, attrs


@attrs
class PathsContainer:
    blup_path = attrib(type=str)
    index_path = attrib(type=str)
    pred_path = attrib(type=str)
    weight_path = attrib(type=str)
    plot_path = attrib(type=str)

    @classmethod
    def from_args(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)
        blup_path = os.path.join(path, "GBLUP")
        index_path = os.path.join(path, "index")
        pred_path = os.path.join(path, "test_pred")
        weight_path = os.path.join(path, 'weight')
        plot_path = os.path.join(path, 'plot')
        for subdir in [blup_path, index_path, pred_path, weight_path, plot_path]:
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
        return cls(blup_path, index_path, pred_path, weight_path, plot_path)
    

def read_anno(directory_path):  
    all_anno_contents = []   
    # 遍历目录中的所有文件  
    for filename in os.listdir(directory_path):  
        # 检查文件扩展名是否为 .snplist  
        if filename.endswith('.snplist'): 
            file_path = os.path.join(directory_path, filename)
            file_contents = np.array(pd.read_csv(file_path, header=None, sep=' '))
            all_anno_contents.append(file_contents)
    return all_anno_contents 
