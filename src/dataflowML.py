from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import linecache
import sys
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Activation
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import load_model

from sklearn.model_selection import train_test_split

import logging
import string
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
file_handler = logging.FileHandler(
    '/home/cry/chengxiao/dataset/tscanc/SARD_119_399/result/log/119_df_result.txt')
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(level=logging.INFO)
logger.addHandler(stream_handler)

NUM_EPOCHS = 50
BATCH_SIZE = 64
HIDDEN_LAYER_SIZE = 64
EMBEDDING_SIZE = 50
MAX_SENTENCE_LENGTH = 50


def appendApiInfo(lenFileVec, info):
    """在info末加入cgd在源文件中具体行的api，用于构建sel数据库.

            Args:
                lenFileVec: 所有cgd的数量
                info: 所有cgd的信息

            Returns:
                None

            Raises:
                IOError: An error occurred accessing the bigtable.Table object.
            """
    logger.info('解析源文件中cgd的api...'.encode('utf-8'))
    inforoot = u'/Users/chengxiao/Desktop/VulDeepecker/ml/resource/CWE-399/source_files/'
    for cdgIndex in range(lenFileVec):
        # print(info[cdgIndex])
        # print("%d %s"%(cdgIndex,info[cdgIndex][1]))
        srcFilePath = inforoot + info[cdgIndex][1]
        # srcFilePath = info[cdgIndex][2]
        srcFileLine = int(info[cdgIndex][3])
        # srcfile = open(srcFilePath, 'r', encoding='utf-8')
        srcFunc = linecache.getline(srcFilePath, srcFileLine).strip()
        alphac = 0
        srcFilePathtmp = srcFilePath
        while ((not srcFunc) and (alphac != 26)):
            srcFilePathtmp = srcFilePath[:srcFilePath.index('.')] + string.ascii_lowercase[alphac] \
                             + srcFilePath[srcFilePath.index('.'):]
            srcFunc = linecache.getline(srcFilePathtmp, srcFileLine).strip()
            linescount = len(linecache.getlines(srcFilePathtmp)) + 1
            srcFileLine = srcFileLine - linescount - 2
            alphac = alphac + 1
        info[cdgIndex].append(srcFunc)
    logger.info('解析api成功！'.encode('utf-8'))


def fixVec(fileIntVec, model, info, tao, type):
    """向量处理.

        对于长度小于tao的向量，如果是前向api，后补0，否则前补0；
        对于长度大雨tao的向量，如果是前向api，删后，否则删前；


        Args:
            fileIntVec: (已将token转成int)codeGadget的tokenIntlist集合 [cg1,cg2,...] cg1:[tk1Int, tk2Int, ...]
            model: 训练模型
            info: 训练集信息（源码位置，函数类型，行号）
            tao: 预定义向量长度
            type: 0 or 1(forw backw)

        Returns:
            None

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
    logger.info('修正向量...'.encode('utf-8'))
    ipt = []
    inforoot = u'/Users/chengxiao/Desktop/VulDeepecker/ml/resource/CWE-399/source_files/'
    for cdgIndex in range(len(fileIntVec)):
        # 通过info读源文件再判断
        cdgInts = fileIntVec[cdgIndex]
        # srcFilePath = inforoot + info[cdgIndex][1]

        # srcFuncType = info[cdgIndex][2]
        # srcFileLine = int(info[cdgIndex][3])
        # srcfile = open(srcFilePath, 'r', encoding='utf-8')
        # srcFunc = info[cdgIndex][4]

        # srcFunc = srcfile.readline(srcFileLine)
        # print(srcFunc)

        if len(cdgInts) < tao:  # 补0
            # print(info[cdgIndex][1])
            fixvec = (tao - len(cdgInts)) * [0]
            # print(fixvec)
            if (type == 1):
                srcFuncType = info[cdgIndex][2]
                # TODO(jumormt): 如何判断api为forward
                if (srcFuncType == 'inputfunc'):  # forward 后补0
                    fileIntVec[cdgIndex].extend(fixvec)
                else:
                    fixvec.extend(cdgInts)
                    fileIntVec[cdgIndex] = fixvec
            else:
                fixvec.extend(cdgInts)
                fileIntVec[cdgIndex] = fixvec

            # print(len(fileIntVec[cdgIndex]))
            # print(fileIntVec[cdgIndex])

        if len(cdgInts) > tao:  # 截断
            # print(info[cdgIndex][1])
            if (type == 1):
                srcFuncType = info[cdgIndex][2]
                # TODO(jumormt): 如何判断api为forward
                if (srcFuncType == 'inputfunc'):  # forward 删后
                    fileIntVec[cdgIndex] = fileIntVec[cdgIndex][:tao]
                else:
                    fileIntVec[cdgIndex] = fileIntVec[cdgIndex][-tao:]
            else:
                fileIntVec[cdgIndex] = fileIntVec[cdgIndex][-tao:]

            # print(len(fileIntVec[cdgIndex]))
            # print(fileIntVec[cdgIndex])
        # print(fileIntVec[cdgIndex])
        # print(len(fileIntVec[cdgIndex]))
    # ----------------------
    logger.info('修正向量成功！'.encode('utf-8'))


def saveVec(picklefile, fileIntVec, target, info):
    """划分数据集并保存数据.

            将数据划分成8：2并使用pickle模块存储,
            x_train 和 x_test存放训练集和测试集向量，
            y_train 和 y_test存放target和info的ziplist


            Args:
                fileIntVec: (已将token转成int)codeGadget的tokenIntlist集合 [cg1,cg2,...] cg1:[tk1Int, tk2Int, ...]
                info: 训练集信息（源码位置，函数类型，行号）
                target: 数据集标记

            Returns:
                None

            Raises:
                IOError: An error occurred accessing the bigtable.Table object.
            """

    logger.info('划分数据集...'.encode('utf-8'))
    # 划分数据集
    Y = list(zip(target, info))
    # X = sequence.pad_sequences(fileIntVec, maxlen=MAX_SENTENCE_LENGTH)
    x_train, x_test, y_train, y_test = train_test_split(fileIntVec, Y, test_size=0.1, random_state=6)
    logger.info('划分数据集成功！'.encode('utf-8'))
    logger.info('保存数据集...'.encode('utf-8'))
    # 保存数据
    try:
        with open(picklefile, 'wb') as pfile:
            pickle.dump(
                {
                    'x_train': x_train,
                    'x_test': x_test,
                    'y_train': y_train,
                    'y_test': y_test,
                },
                pfile, pickle.HIGHEST_PROTOCOL
            )
    except Exception as e:
        logger.info('Unable to save data to', picklefile, ':', e)
        raise

    logger.info('Data cached in pickle file.')


def loadVec(picklefile):
    """读取序列化后的向量数据.

                读取训练集和测试集

                Args:
                    picklefile: picklefile的位置

                Returns:
                    x_train: 训练集
                    x_test: 测试集
                    y_train: 训练集的(label,info)list
                    y_test: 测试集的(label, info)list

                Raises:
                    IOError: An error occurred accessing the bigtable.Table object.

                """
    logger.info('载入数据集...'.encode('utf-8'))
    try:
        with open(picklefile, 'rb') as f:
            pickleData = pickle.load(f)
            x_train = pickleData['x_train']
            y_train = pickleData['y_train']
            x_test = pickleData['x_test']
            y_test = pickleData['y_test']
            del pickleData
    except Exception as e:
        logger.info('Unable to load data:', picklefile, ':', e)
        raise
    logger.info('载入数据集成功！'.encode('utf-8'))
    return x_train, y_train, x_test, y_test


# # TODO(Jumormt): 建立双向lstm模型
# def createModel(inputLen, maxIndex, wordSize):
#     """创建神经网络.
#
#             创建双向lstm神经网络，包括输入层，循环层，输出层等。
#
#             Args:
#                 inputLen: 输入向量维度
#                 maxIndex：词向量最大index
#                 wordSize: 词向量维度（暂时为1）
#
#             Returns:
#                 model:网络模型
#
#             Raises:
#                 IOError: An error occurred accessing the bigtable.Table object.
#             """
#     logger.info('创建blstm神经网络...')
#     inpt = Input(shape=(inputLen,), dtype='int32')
#     # embedded = Embedding(maxIndex + 1, wordSize, input_length=inputLen, mask_zero=True)(inpt)
#     blstm = Bidirectional(LSTM(64, input_shape=(inputLen, 1), return_sequences=True), merge_mode='sum')(inpt)
#     output = TimeDistributed(Dense(1, activation='sigmoid'))(blstm)
#     model = Model(input = inpt, output = output)
#     model.compile(loss='binary_crossentropy', optimizer = 'adamax', metrics = ['accuracy'])
#     logger.info('创建blstm神经网络成功！')
#     return model

# TODO(Jumormt): 建立双向lstm模型
def createModel(inputLen, maxIndex, wordSize):
    """创建神经网络.

            创建双向lstm神经网络，包括输入层，循环层，输出层等。

            Args:
                inputLen: 输入向量维度
                maxIndex：词向量最大index
                wordSize: 词向量维度（暂时为1）

            Returns:
                model:网络模型

            Raises:
                IOError: An error occurred accessing the bigtable.Table object.
            """
    logger.info('创建blstm神经网络...'.encode('utf-8'))
    model = Sequential()
    # model.add(Embedding(maxIndex + 1, wordSize, input_length=inputLen, mask_zero=True))
    model.add(Embedding(maxIndex + 1, wordSize, input_length=inputLen))
    model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['binary_accuracy'])
    logger.info('创建blstm神经网络成功！'.encode('utf-8'))
    return model


def main():
    # logging.basicConfig(filename=u'~/ml/out/log/logger.log', level=logging.INFO)

    # 获取训练集------------------------------------
    # sentences = Text8Corpus(u"~/ml/resource/text8")

    # sentences = [['first', 'sentence'], ['second', 'sentence']]
    logger.info('载入文本数据...'.encode('utf-8'))
    resourcepath = u"~/ml/resource/"
    # datapath = u"~/ml/resource/test_error_case.txt"
    # datapath = u"~/ml/resource/cwe119_cgd_result.txt"
    # datapath = u"~/ml/resource/test_output.txt"
    # datapath = u"/Users/chengxiao/Desktop/VulDeepecker/ml/resource/cwe/cwe399/cwe399_cgd_result.txt"
    # datapath = u"/Users/chengxiao/Desktop/VulDeepecker/ml/resource/cwe/cwe119/cwe119_cgd_result.txt"
    # datapath = u"/Users/chengxiao/Desktop/VulDeepecker/资料/project/CGDSymbolization/src/main/resources/output1.txt"
    # datapath = u"/Users/chengxiao/Downloads/119/sard_sym.txt"
    # datapath = u"/Users/chengxiao/Downloads/CWE-840/sard_sym.txt"
    # datapath = u"/home/cry/chengxiao/dataset/VulDeepeckerDB/840_sym.txt"
    datapath = u"/home/cry/chengxiao/dataset/VulDeepeckerDB/cwe119_cgd_result.txt"
    sentences = LineSentence(datapath)

    fileVec = []  # codeGadget的tokenlist集合 [cg1,cg2,...] cg1:[tk1, tk2, ...]
    info = []  # codeGadget的信息集合 [序号, src位置, 函数类型, 行数, api]
    target = []  # codeGadget的标记

    cdgVec = []  # 待加入fileVec的当前cdg的tokenlist
    currentCdgVec = []  # 当前cdg的token集合，二维数组
    for line in sentences:
        """
        functype = ['Off_by_One_Error_in_Methods', 'Buffer_Overflow_fgets', 'cppfunc',
                    'Buffer_Overflow_LowBound', 'MultiByte_String_Length', 'inputfunc',
                    'Buffer_Overflow_boundedcpy', 'Format_String_Attack',
                    'Buffer_Overflow_scanf', 'String_Termination_Error',
                    'Buffer_Overflow_Indexes', 'cfunc', 'Buffer_Overflow_unbounded',
                    'API', 'Buffer_Overflow_cpycat', 'Missing_Precision']
        """

        currentCdgVec.append(line)
        if (line[0] == "---------------------------------"):
            currentCdgVec.pop()  # 移除分隔符
            currentCdgTarget = currentCdgVec.pop()
            target.append(int(currentCdgTarget[0]))  # target
            # print(currentCdgTarget[0])
            infoLine = currentCdgVec[0]
            # print(infoLine)
            infoVec = []
            infoVec.append(infoLine[0])
            srcLocList = infoLine[:len(infoLine) - 2][1:]
            srcLoc = " ".join(srcLocList)  # 将src位置合并为一个字符串
            infoVec.append(srcLoc)  # src位置
            # infoVec.append(infoLine[len(infoLine)-3]) #函数类型
            infoVec.append(infoLine[len(infoLine) - 2])  # 行号
            infoVec.append(infoLine[len(infoLine) - 1])  # api
            info.append(infoVec)
            # print("before")
            # print(currentCdgVec[0])
            del currentCdgVec[0]
            # print("after")
            # print(currentCdgVec[0])
            # print(infoVec)
            for tokenLine in currentCdgVec:
                # print(tokenLine)
                cdgVec.extend(tokenLine)
            # print(cdgVec)
            fileVec.append(cdgVec)
            currentCdgVec = []
            cdgVec = []

        # if (len(line) == 4 and line[0].isdigit()
        #    and line[3].isdigit()):
        # info.append(" ".join(line))
        #    info.append(line)
        #    continue

        # if (len(line) == 1 and (line[0] == "0" or line[0] == "1")):
        #    fileVec.append(cdgVec)
        #    target.append(int(line[0]))
        #    cdgVec = []
        #    continue

        # if (line[0] == "---------------------------------"):
        #    continue

        # cdgVec.extend(line)
    # print(info)
    # appendApiInfo(len(fileVec), info)
    logger.info('载入文本数据成功！'.encode('utf-8'))
    # f = open("/home/centos/ml/resource/model2output.txt","w")
    # print(fileVec,file = f)
    # print(info,file = f)
    # print(target,file =f)
    # for i in range(1,10):
    #    print(fileVec[i])

    # 训练模型------------------------------------
    logger.info('训练word2vec模型中...'.encode('utf-8'))
    # outpm = u"/Users/chengxiao/Desktop/VulDeepecker/ml/resource/model/word2vec_1102_test.model"
    outpm = u"/home/cry/chengxiao/dataset/VulDeepeckerDB/model/word2vec_1102_test.model"
    model = Word2Vec(sentences, min_count=0, size=1)
    logger.info('词袋模型训练成功！'.encode('utf-8'))
    model.save(outpm)

    # model = Word2Vec.load(outpm)

    # 构建向量------------------------------------
    # wordVc = model.wv.index2entity
    # mp = {}
    # for i in range(len(wordVc)):
    #     mp[wordVc[i]] = i
    #
    #
    # print(model.wv.index2entity.index("="))
    # print(model.wv.word_vec("="))
    # print(mp["="])
    # wordDic = model.wv.vocab.keys()
    # print(type(wordDic))
    logger.info('构建向量数据库...'.encode('utf-8'))
    i = 0
    fileIntVec = []  # (已将token转成int)codeGadget的tokenIntlist集合 [cg1,cg2,...] cg1:[tk1Int, tk2Int, ...]
    for cdgs in fileVec:
        cdgInts = []
        for token in cdgs:
            if token in model.wv.vocab:
                cdgInts.append(model.wv.vocab[token].index + 1)
            else:
                cdgInts.append(0)
        # print(cdgInts)
        fileIntVec.append(cdgInts)

    fixVec(fileIntVec, model, info, MAX_SENTENCE_LENGTH, 1)
    for i in range(1, 6):
        print(i)
        print(fileVec[i])
        print(fileIntVec[i])
        sys.stdout.flush()
    logger.info('构建向量数据库成功！'.encode('utf-8'))
    # ----------------------

    # 保存向量化数据
    picklefile = u"/home/cry/chengxiao/dataset/VulDeepeckerDB/vecdata/vecdata.pickle"
    # saveVec(picklefile, fileIntVec, target, info)

    # 载入向量化数据
    # x_train, y_train_r, x_test, y_test_r = loadVec(picklefile)
    # y_train = [i[0] for i in y_train_r]
    # trainInfo = [i[1] for i in y_train_r]
    # y_test = [i[0] for i in y_test_r]
    # test_info = [i[1] for i in y_test_r]

    # 载入picklefile
    # try:
    #     with open(picklefile, 'rb') as f:
    #         pickleData = pickle.load(f)
    #         x_train = pickleData['x_train']
    #         y_train = pickleData['y_train']
    #         x_test = pickleData['x_test']
    #         y_test = pickleData['y_test']
    #         del pickleData
    # except Exception as e:
    #     print('Unable to load data:', picklefile, ':', e)
    #     raise

    # 创建神经网络------------------------------------

    maxindex = len(model.wv.index2entity)

    kf = KFold(n_splits=10, shuffle=True)
    tprList = list()
    fprList = list()
    fnrList = list()
    f1List = list()
    AUCList = list()
    accuracyList = list()
    kfcount = 1
    fileIntVec = np.array(fileIntVec)
    target = np.array(target)
    for train_idx, test_idx in kf.split(fileIntVec):
        logger.info("split: {}".format(kfcount))
        kfcount = kfcount + 1
        x_train, y_train = fileIntVec[train_idx], target[train_idx]
        x_test, y_test = fileIntVec[test_idx], target[test_idx]

        blstmCgdModel = createModel(MAX_SENTENCE_LENGTH, maxindex, EMBEDDING_SIZE)
        print(blstmCgdModel.metrics_names)

        blstmCgdModel.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=2)
        # blstmCgdModel.save('/Users/chengxiao/Desktop/VulDeepecker/ml/resource/model/model-total-1105-test.h5')

        # blstmCgdModel = load_model('model-total-1102.h5')
        # print(type(x_test))
        # y_result = blstmCgdModel.predict_classes(x_train)
        # metricss = blstmCgdModel.evaluate(x_test, y_test, verbose=0)
        y_result = blstmCgdModel.predict_classes(x_test)
        # print('f1: %.8f' % metrics.f1_score(y_test, y_result))
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(y_result)):
            if (y_test[i] == 1):
                if (y_result[i] == 1):
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:
                if (y_result[i] == 1):
                    FP = FP + 1
                else:
                    TN = TN + 1
        TPR = round(TP / (TP + FN), 10)
        logger.info("tpr: {}".format(TPR))
        FPR = round(FP / (FP + TN), 10)
        logger.info("fpr: {}".format(FPR))
        FNR = round(FN / (TP + FN), 10)
        logger.info("fnr: {}".format(FNR))
        if (TP + FP != 0):
            P = round(TP / (TP + FP), 10)
            logger.info("f1: {}".format(round(2 * P * TPR / (P + TPR), 10)))
        accuracy = metrics.accuracy_score(y_test, y_result)
        logger.info("accuracy: {}".format(accuracy))
        AUC = metrics.roc_auc_score(y_test, y_result)
        logger.info('AUC: %.8f' % AUC)
        tprList.append(TPR)
        fprList.append(FPR)
        fnrList.append(FNR)
        accuracyList.append(accuracy)
        AUCList.append(AUC)
        # ji = int(1)
        # for i in range(len(y_result)):
        #    print("%d %d %d"%(i,y_result[i][0],y_train[i]))
        # for item in y_result:
        #    print("%d %s" % (ji,item[0]))
        #    ji=ji+1
        # print(y_result)
        # logger.info(metricss)

    logger.info("tpr: {}".format(np.mean(tprList)))
    logger.info("fpr: {}".format(np.mean(fprList)))
    logger.info("fnr: {}".format(np.mean(fnrList)))
    logger.info("accuracy: {}".format(np.mean(accuracyList)))
    logger.info("AUC: {}".format(np.mean(AUCList)))

    print("end")


CWE119SEL = ["cin", "getenv", "getenv_s", "wgetenv", "wgetenv_s", "catgets", "gets",
             "getchar", "getc", "getch", "getche", "kbhit", "stdin", "getdlgtext", "getpass",
             "scanf", "fscanf", "vscanf", "vfscanf", "istream.get", "istream.getline",
             "istream.peek", "istream.read*", "istream.putback", "streambuf.sbumpc",
             "streambuf.sgetc", "streambuf.sgetn", "streambuf.snextc", "streambuf.sputbackc",
             "SendMessage", "SendMessageCallback", "SendNotifyMessage", "PostMessage",
             "PostThreadMessage", "recv", "recvfrom", "Receive", "ReceiveFrom", "ReceiveFromEx",
             "Socket.Receive*", "memcpy", "wmemcpy", "memccpy", "memmove", "wmemmove", "memset",
             "wmemset", "memcmp", "wmemcmp", "memchr", "wmemchr", "strncpy", "strncpy*", "lstrcpyn",
             "tcsncpy*", "mbsnbcpy*", "wcsncpy*", "wcsncpy", "strncat", "strncat*", "mbsncat*",
             "wcsncat*", "bcopy", "strcpy", "lstrcpy", "wcscpy", "tcscpy", "mbscpy", "CopyMemory",
             "strcat", "lstrcat", "lstrlen", "strchr", "strcmp", "strcoll", "strcspn", "strerror",
             "strlen", "strpbrk", "strrchr", "strspn", "strstr", "strtok", "strxfrm", "readlink",
             "fgets", "sscanf", "swscanf", "sscanf s", "swscanf s", "printf", "vprintf", "swprintf",
             "vsprintf", "asprintf", "vasprintf", "fprintf", "sprint", "snprintf", "snprintf*",
             "snwprintf*", "vsnprintf", "CString.Format", "CString.FormatV", "CString.FormatMessage",
             "CStringT.Format", "CStringT.FormatV", "CStringT.FormatMessage",
             "CStringT.FormatMessageV", "syslog", "malloc", "Winmain", "GetRawInput*",
             "GetComboBoxInfo", "GetWindowText", "GetKeyNameText", "Dde*", "GetFileMUI*",
             "GetLocaleInfo*", "GetString*", "GetCursor*", "GetScroll*", "GetDlgItem*", "GetMenuItem*"]

CWE399SEL = ["free", "delete", "new", "malloc", "realloc", "calloc", "alloca",
             "strdup", "asprintf", "vsprintf", "vasprintf", "sprintf", "snprintf",
             "snprintf", "snwprintf", "vsnprintf"]

if __name__ == '__main__':
    main()
