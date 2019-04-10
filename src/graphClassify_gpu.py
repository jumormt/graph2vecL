import csv
# https://blog.csdn.net/john_xyz/article/details/79208564
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
# import torch
# from torch_geometric.data import Data

# data.x: Node feature matrix with shape [num_nodes, num_node_features]
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
# data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
# data.y: Target to train against (may have arbitrary shape)
# data.pos: Node position matrix with shape [num_nodes, num_dimensions]

import os
import json

import argparse

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score
from sklearn import svm
import numpy as np
from sklearn.model_selection import ShuffleSplit

from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.ensemble import EasyEnsemble  # 简单集成方法EasyEnsemble

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

from sklearn import preprocessing

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold

import logging
import sys
import time

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(
    '../resources/result/test787_xgboost_gpu_result.txt')
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(level=logging.INFO)
logger.addHandler(stream_handler)


def getGraphs(inputpath, inputCSVPath):
    '''得到graphs，key为图向量x和标记target

    :param inputpath: graphs-json
    :param inputCSVPath: graphs-vec
    :return: graph
    '''

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

    # get x
    with open(inputCSVPath, encoding="utf-8") as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            name = row[0]
            vec = row[1:]
            graphs[name + ".json"]['x'] = vec
    return graphs


def getXY(graphs):
    '''
    得到经过均衡处理后的xy，并对x进行预处理
    :param graphs: getGraph得到的图
    :return: X，Y-list
    '''
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
    logger.info(sorted(Counter(y_resampled).items()))
    # print(sorted(Counter(y_resampled).items()))

    # rus = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = rus.fit_sample(X, Y)
    # logger.info(sorted(Counter(y_resampled).items()))

    # 预处理 (X-mean)/std  计算时对每个属性/每列分别进行。
    # 将数据按期属性（按列进行）减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。
    scaler = preprocessing.StandardScaler().fit(X_resampled)
    X_train_transformed = scaler.transform(X_resampled)

    return X_train_transformed, y_resampled


def svm_cross_validation(train_x, train_y):
    '''
    svm调参
    :param train_x:
    :param train_y:
    :return:
    '''
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=8, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    logger.info("SVM best param:")
    for para, val in list(best_parameters.items()):
        # print(para, val)
        logger.info(str(para) + " " + str(val))
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    # model.fit(train_x, train_y)
    return model


def KNN_cross_validation(train_x, train_y):
    '''

    :param train_x:
    :param train_y:
    :return:
    '''

    clf = KNeighborsClassifier(n_neighbors=3)
    param_grid = {'n_neighbors': list(range(1, 19)),
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    logger.info("KNN best param:")
    for para, val in list(best_parameters.items()):
        # print(para, val)
        logger.info(str(para) + " " + (val))
    clf = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'],
                               metric=best_parameters['metric'])

    return clf


def LR_cross_validation(train_x, train_y):
    '''
    逻辑回归
    :param train_x:
    :param train_y:
    :return:
    '''
    clf = LogisticRegression()
    # param_grid = {'penalty': ['l1', 'l2'],
    #               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    #               'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
    param_grid = [{'penalty': ['l1', 'l2'],
                   'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                   'solver': ['liblinear', 'saga']
                   },
                  {'penalty': ['l2'],
                   'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                   'solver': ['lbfgs', 'newton-cg', 'sag']
                   }]

    grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    logger.info("LR best param:")
    for para, val in list(best_parameters.items()):
        # print(para, val)
        logger.info(str(para) + " " + str(val))
    clf = LogisticRegression(penalty=best_parameters['penalty'], solver=best_parameters['solver'],
                             C=best_parameters['C'])

    return clf


def NaiveBayes_cross_validation(train_x, train_y):
    '''
    朴素贝叶斯
    :param train_x:
    :param train_y:
    :return:
    '''
    clf = GaussianNB()
    # param_grid={'alpha':[1, 0.1, 0.01, 0.001, 0.0001]}
    #
    # grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1, cv=5)
    # grid_search.fit(train_x, train_y)
    # best_parameters = grid_search.best_estimator_.get_params()
    # logger.info("NAiveBayes best param:")
    # for para, val in list(best_parameters.items()):
    #     # print(para, val)
    #     logger.info(str(para) + " " + str(val))
    # clf = GaussianNB()

    return clf


def DecisionTree_cross_validation(train_x, train_y):
    '''
    决策树
    :param train_x:
    :param train_y:
    :return:
    '''
    clf = DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_features': range(3, 9, 1)
                  }

    grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    logger.info("DesisionTree best param:")
    for para, val in list(best_parameters.items()):
        # print(para, val)
        logger.info(str(para) + " " + str(val))
    clf = DecisionTreeClassifier(criterion=best_parameters['criterion'], splitter=best_parameters['splitter'])

    return clf


def RandomForest_cross_validation(train_x, train_y):
    '''
    随机森林
    :param train_x:
    :param train_y:
    :return:
    '''
    clf = RandomForestClassifier(random_state=0)
    # param_grid = {'criterion': ['gini', 'entropy'],
    #               'n_estimators': [10, 100, 200, 500, 700, 1000],
    #               'max_features': ['auto', 'sqrt', 'log2']}
    param_grid = {'criterion': ['gini'],
                  'n_estimators': [680, 700, 720],
                  'max_depth': [30, 50, 100],
                  'max_features': ['log2']}

    grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1, cv=5)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    logger.info("DesisionTree best param:")
    for para, val in list(best_parameters.items()):
        # print(para, val)
        logger.info(str(para) + " " + str(val))
    clf = RandomForestClassifier(random_state=0, criterion=best_parameters['criterion'],
                                 n_estimators=best_parameters['n_estimators'],
                                 max_features=best_parameters["max_features"],
                                 max_depth=best_parameters["max_depth"])

    return clf


def gradient_boosting_classifier_cross_validation(train_x, train_y):
    '''
    GBDT(Gradient Boosting Decision Tree) Classifier
    :param train_x:
    :param train_y:
    :return:
    '''

    # logger.info("gradient_boosting best param:")
    clf = GradientBoostingClassifier(max_features=11, min_samples_leaf=40, n_estimators=290, max_depth=13,
                                     min_samples_split=80)

    # param1 = {'n_estimators': range(200, 301, 10)}
    # grid_search = GridSearchCV(clf, param1, n_jobs=8, verbose=1, cv=5)
    # grid_search.fit(train_x, train_y)
    # best_parameters1 = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters1.items()):
    #     # print(para, val)
    #     logger.info(str(para) + " " + str(val))

    # clf = GradientBoostingClassifier(n_estimators=best_parameters1['n_estimators'])
    # param_test2 = {'max_depth': range(5, 16, 2), 'min_samples_split': range(200, 1001, 200)}
    # grid_search = GridSearchCV(clf, param_test2, n_jobs=8, verbose=1, cv=5)
    # grid_search.fit(train_x, train_y)
    # best_parameters2 = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters2.items()):
    #     # print(para, val)
    #     logger.info(str(para) + " " + str(val))
    #
    # clf = GradientBoostingClassifier(min_samples_split=best_parameters2['min_samples_split'],
    #                                  max_depth=best_parameters2['max_depth'],
    #                                  n_estimators=best_parameters1['n_estimators'])
    # param_test3 = {'min_samples_split': range(1000, 2100, 200), 'min_samples_leaf': range(30, 71, 10)}
    # grid_search = GridSearchCV(clf, param_test3, n_jobs=8, verbose=1, cv=5)
    # grid_search.fit(train_x, train_y)
    # best_parameters3 = grid_search.best_estimator_.get_params()
    # for para, val in list(best_parameters3.items()):
    #     # print(para, val)
    #     logger.info(str(para) + " " + str(val))
    #
    # clf = GradientBoostingClassifier(min_samples_split=best_parameters2['min_samples_split'],
    #                                  max_depth=best_parameters2['max_depth'],
    #                                  n_estimators=best_parameters1['n_estimators'])
    #
    # param_test4 = {'max_features': range(7, 20, 2)}
    return clf


def xgboost_cross_validation(train_x, train_y):
    '''
    xgboost
    :param train_x:
    :param train_y:
    :return:
    '''
    from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
    import xgboost as xgb
    from scipy.stats import uniform, randint
    # xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(300, 400),  # default 100
        "subsample": uniform(0.6, 0.4)
    }


    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=300)
    # cv_params = {'n_estimators': [800, 900, 1000]}
    # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    # cv_params = {
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    # cv_params = {
    #     'subsample': [i / 10.0 for i in range(6, 10)],
    #     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    # }
    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # cv_params = {'n_estimators': [1000, 1500, 2000]}
    # optimized_GBM = GridSearchCV(estimator=xgb_model, param_grid=cv_params, cv=3, verbose=1, n_jobs=8)
    # optimized_GBM.fit(train_x, train_y)
    # best_para = optimized_GBM.best_estimator_.get_params()
    # for para, val in list(best_para.items()):
    #     # print(para, val)
    #     logger.info(str(para) + " " + str(val))
    # model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=best_para['n_estimators'])




    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=5, verbose=1,
                                n_jobs=8, return_train_score=True)

    search.fit(train_x, train_y)
    best_parameters = search.best_estimator_.get_params()
    logger.info("xgboost best param:")
    for para, val in list(best_parameters.items()):
        # print(para, val)
        logger.info(str(para) + " " + str(val))
    model = xgb.XGBClassifier(subsample=best_parameters["subsample"], max_depth=best_parameters["max_depth"],
                              n_estimators=best_parameters["n_estimators"],
                              gamma=best_parameters["gamma"], learning_rate=best_parameters["learning_rate"],
                              colsample_bytree=best_parameters["colsample_bytree"], objective="binary:logistic",
                              random_state=42)
    return model


def get_xgboost_params(X, Y):
    '''

    :param X:
    :param Y:
    :return:
    '''
    import numpy as np
    import xgboost
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import mean_squared_error
    cv = StratifiedKFold(n_splits=5)
    SCORE_MIN = False
    best_score = 0
    best_iter = None
    best_params = None
    # param_grid = [
    #     {'silent': [1],
    #      'eval_metric': ['error'],
    #      'tree_method': ['gpu_hist'],
    #      'objective': ['binary:logitraw'],
    #      'max_depth': [5, 7],
    #      'num_round': [1000],
    #      'subsample': [0.2, 0.4, 0.6],
    #      'colsample_bytree': [0.3, 0.5, 0.7],
    #      }
    # ]
    param_grid = [
        {'silent': [1],
         'eval_metric': ['error'],
         'tree_method': ['gpu_hist'],
         'objective': ['binary:logistic'],
         'num_round': [967],
         'max_depth':[7],
         'min_child_weight':[1],
         'gamma': [i / 10.0 for i in range(0, 5)]
         }
    ]

    num_round = None
    # Hyperparmeter grid optimization
    for params in ParameterGrid(param_grid):
        print(params)
        # Determine best n_rounds
        xgboost_rounds = []
        for i, (train, test) in enumerate(cv.split(X, Y)):
            xg_train = xgb.DMatrix(X[train], label=Y[train])
            xg_test = xgb.DMatrix(X[test], label=Y[test])

            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            num_round = params['num_round']
            xgclassifier = xgboost.train(params, xg_train, num_round,
                                         watchlist,
                                         early_stopping_rounds=100)
            xgboost_rounds.append(xgclassifier.best_iteration)

        num_round = int(np.mean(xgboost_rounds))
        print('The best n_rounds is %d' % num_round)
        # Solve CV
        rmsle_score = []
        for i, (train, test) in enumerate(cv.split(X, Y)):
            xg_train = xgb.DMatrix(X[train], label=Y[train])
            xg_test = xgb.DMatrix(X[test], label=Y[test])
            y_test = Y[test]
            watchlist = [(xg_train, 'train'), (xg_test, 'test')]

            xgclassifier = xgboost.train(params, xg_train, num_round)
            # res = xgclassifier.eval(xgb.DMatrix(X[test], label=Y[test]))
            # print("2222222222",type(res))

            # predict
            from sklearn import metrics
            predicted_results = xgclassifier.predict(xg_test)
            y_pred = (predicted_results >= 0.5)*1

            print(
            'ACC: %.4f' % metrics.accuracy_score( y_test,y_pred))
            # print('AUC: %.4f' % metrics.roc_auc_score(y_test, y_pred))
            #
            # print('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
            # print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
            # print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))
            # print(metrics.confusion_matrix(y_test, y_pred))

            # rmsle_score.append(np.sqrt(mean_squared_error(y_test, predicted_results)))
            rmsle_score.append(metrics.accuracy_score(y_test,y_pred))

        if SCORE_MIN:
            if best_score > np.mean(rmsle_score):
                print(np.mean(rmsle_score))
                print('new best')
                best_score = np.mean(rmsle_score)
                best_params = params
                best_iter = num_round
        else:
            if best_score < np.mean(rmsle_score):
                print(np.mean(rmsle_score))
                print('new best')
                best_score = np.mean(rmsle_score)
                best_params = params
                best_iter = num_round

    # Solution using best parameters
    logger.info('best params: %s' % best_params)
    logger.info('best score: %f' % best_score)
    logger.info('best num_round: %s' % best_iter)
    return best_params, best_iter

def main(args):
    graphs = getGraphs(args.input_json_path, args.input_csv_path)
    X, Y = getXY(graphs)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # scoring = ['precision_macro', 'recall_macro']

    # clf.fit(X_train, y_train)
    # a = clf.predict(X_test)
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X, Y)
    # sorted(Counter(y_resampled).items())

    # clf = svm_cross_validation(X, Y)
    # clf = KNN_cross_validation(X, Y)
    # clf = DecisionTree_cross_validation(X, Y)
    # clf = NaiveBayes_cross_validation(X, Y)
    # clf = RandomForestClassifier(criterion='gini',max_depth=50, max_features='log2', n_estimators=700)
    # clf = gradient_boosting_classifier_cross_validation(X, Y)
    # clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_features=6)
    # clf = SVC(C=1000, gamma=0.01)
    # clf = GradientBoostingClassifier(max_features=11, min_samples_leaf=40, n_estimators=290, max_depth=13,
    #                                  min_samples_split=80)
    # clf = xgboost_cross_validation(X, Y)
    clf = xgb.XGBClassifier(subsample=0.8324952885690449, n_estimators=137, colsample_bytree=0.772722919724524,
                            gamma=0.05741841236960177, max_depth=5,
                            learning_rate=0.21318601273248977)
    cv = StratifiedKFold(n_splits=5)


    # param = {
    #     'objective':'binary:logitraw',
    #     'eval_metric':['auc','map','error'],
    #     'tree_method':'gpu_hist',
    #     'silent':1,
    #     'max_depth':5,
    #     'learning_rate':0.1,
    #     'n_estimators':1000,
    #     'min_child_weight':1,
    #     'gamma':0,
    #     'subsample':0.8,
    #     'colsample_bytree':0.8,
    #     'scale_pos_weight':1
    # }

    tmp = time.time()
    best_params, num_round = get_xgboost_params(X,Y)
    boost_time = time.time() - tmp
    logger.info("get xgboost best params Time {}".format(str(boost_time)))
    best_params['num_round'] = num_round
    best_params['eval_metric']= ['auc', 'map', 'error']

    for i, (train, test) in enumerate(cv.split(X, Y)):
        dtrain = xgb.DMatrix(X[train], label=Y[train])
        tmp = time.time()
        bst = xgb.train(best_params, dtrain, num_round)
        boost_time = time.time() - tmp
        res = bst.eval(xgb.DMatrix(X[test], label=Y[test]))
        logger.info("Fold {}: {}, Boost Time {}".format(i, res, str(boost_time)))
        del bst

    # gpu_res = {}
    # xgb.train(param, X_train, 1000, evals=[(X_test, "test")],
    #           evals_result=gpu_res)

    # scores = cross_val_score(clf, X, Y, cv=cv)
    # logger.info("XGB Accuracy:")
    # for i in range(len(scores)):
    #     # print("Accuracy%d: %0.5f" % (i, scores[i]))
    #     logger.info("Accuracy%d: %0.5f" % (i, scores[i]))
    #
    # print("Accuracy average: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    # logger.info(
    #     "Accuracy average: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    #
    # print("end")


def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """

    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--input-json-path",
                        nargs="?",
                        # default="/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/result",
                        default="/home/cry/chengxiao/dataset/SARD.2019-02-28-22-07-31/result_sym",
                        help="Input folder with jsons.")
    parser.add_argument("--input-csv-path",
                        nargs="?",
                        default="../features/test-787.csv",
                        help="Input csv file which contains graphvecs.")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs. Default is 10.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parameter_parser()
    main(args)
