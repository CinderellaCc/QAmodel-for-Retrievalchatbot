# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time
from jiebaSegment import *
from sentenceSimilarity import SentenceSimilarity

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese


def read_corpus():
    qList = []
    qList_kw = []   # 问题的关键词列表
    aList = []
    data = pd.read_excel('data/QA.xls', header=None)
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(t[2])
        qList_kw.append(seg.cut(t[2]))
        aList.append(t[3])
    return qList_kw, qList, aList


def plot_words(wordList):
    fDist = FreqDist(wordList)
    # print(fDist.most_common())
    print("单词总数: ", fDist.N())
    print("不同单词数: ", fDist.B())
    fDist.plot(10)


if __name__ == '__main__':
    # 设置外部词
    seg = Seg()
    seg.load_userdict('userdict/userdict.txt')  # 添加自己的词库到默认词库中
    # 读取数据
    _, questionList, answerList = read_corpus()
    # 初始化模型
    ss = SentenceSimilarity(seg)  # 设置self.reg属性
    ss.set_sentences(questionList)  # 设置self.sentences属性，列表类型，列表中每个值为Sentence对象
    ss.TfidfModel()  # tfidf模型
    # ss.LsiModel()         # lsi模型
    # ss.LdaModel()         # lda模型

    while True:
        question = input("请输入问题(q退出): ")
        if question == 'q':
            break
        time1 = time.time()
        question_k = ss.similarity_k(question, 5)
        print("亲，我们给您找到的答案是： {}".format(answerList[question_k[0][0]]))
        for idx, score in zip(*question_k):
            print("same questions： {},                score： {}".format(questionList[idx], score))
        time2 = time.time()
        cost = time2 - time1
        print('Time cost: {} s'.format(cost))
