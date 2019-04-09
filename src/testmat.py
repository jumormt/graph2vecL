# import mat4py
# a = mat4py.loadmat('/Users/chengxiao/Downloads/FGSD-master/MATLAB/data/G_mutag.mat')
# strr = '/Users/chengxiao/Downloads/graph2vec-master/dataset_test/opcua_simpletypes.c.exm_3_sym.json'
# split_ = strr.split('/')[-1]
# strip = split_[:-5]
# print("end")
import os
import json
from collections import Counter

inputpath = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/result"
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

y = [graphs[graph]['target'] for graph in graphs]
print(sorted(Counter(y).items()))