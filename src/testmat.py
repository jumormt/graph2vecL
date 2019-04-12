# import mat4py
# a = mat4py.loadmat('/Users/chengxiao/Downloads/FGSD-master/MATLAB/data/G_mutag.mat')
# strr = '/Users/chengxiao/Downloads/graph2vec-master/dataset_test/opcua_simpletypes.c.exm_3_sym.json'
# split_ = strr.split('/')[-1]
# strip = split_[:-5]
# print("end")
import os
import json
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.ensemble import EasyEnsemble  # 简单集成方法EasyEnsemble

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
#
inputpath = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/noerror/result"
graphs = dict()

# get target
for dirpath, dirnames, filenames in os.walk(inputpath):
    # for dir in dirnames:
    #     fulldir = os.path.join(dirpath,dir)
    #     print(fulldir)

    for file in filenames:  # 遍历完整文件
        fullpath = os.path.join(dirpath, file)
        # print (fullpath)
        with open(fullpath, 'r', encoding="utf-8") as f:
            curjson = json.load(f)
            if ("target" not in curjson.keys()):
                continue
            nodeStrList = curjson["nodes"]

            target = curjson["target"]
            edgeList = curjson["edges"]
            graphs[file] = dict()
            graphs[file]["target"] = target
            # graphs[file]["edgeList"] = edgeList
            # graphs[file]["nodeStrList"] = nodeStrList
X = list()
Y = list()

for graph in graphs:
    X.append(graphs[graph]['x'])
    Y.append(graphs[graph]['target'])

X = np.array(X).astype('float64')
Y = np.array(Y)

# 结合采样
# https://blog.csdn.net/kizgel/article/details/78553009
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_sample(X, Y)
# X_resampled, y_resampled = X,Y
print(sorted(Counter(y_resampled).items()))
y = [graphs[graph]['target'] for graph in graphs]
print(sorted(Counter(y).items()))
# print(round(-0.222))