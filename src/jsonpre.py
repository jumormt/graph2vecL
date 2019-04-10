import json
import os

# inputpath = "/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/result"
inputpath = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/cut_result_sym/"
# outputjson = "/Users/chengxiao/Downloads/graph2vec-master/dataset_test/"
outputjson = "/Users/chengxiao/Downloads/SARD.2019-02-28-22-07-31/cut_result_sym_pre/"

def main():
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

                #target = curjson["target"]
                edges = curjson["edges"]
                features = dict()
                for i in range(len(nodeStrList)):
                    features[str(i)] = nodeStrList[i]
                jsonob = dict()
                jsonob["edges"] = edges
                jsonob["features"] = features

                outputfilename = file
                # outputj = json.dumps(jsonob, sort_keys=True, indent=4, separators=(',', ': '))
                outputj = json.dumps(jsonob)
                with open(outputjson+outputfilename, 'w', encoding="utf-8") as f:
                    f.write(outputj)
                    print("writing"+outputfilename)



if __name__ == '__main__':
    main()
