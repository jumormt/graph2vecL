import csv

inputCSVPath = "/Users/chengxiao/Downloads/graph2vec-master/features/test2.csv"
graphVec = dict()
with open(inputCSVPath, encoding="utf-8") as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        name = row[0]
        vec = row[1:]
        graphVec[name+".json"] = dict()
        graphVec[name + ".json"]['x'] = vec

print(graphVec)