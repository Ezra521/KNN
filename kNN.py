"""
Created on Dec 12, 2017
kNN: k Nearest Neighbors

Input:      inX: 与现有的数据集进行比较的向量 (1xN)
            dataSet: 已知的m个数据集 (NxM)
            labels: 数据集的标签m维 (1xM 向量)
            k: 用于比较的邻居的数量 (必须是奇数，一般小于20)

Output:     最接近的类标签

@author: Ezra
@email:zgahwuqiankun@qq.com
参考机器学习实践
这个例子比较简单，数据集一共四个点
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group ,labels

group,labels = createDataSet()


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #计算dataSet里总共有多少样本
    #tile()是扩展函数，就是按照x轴或者y轴把数据复制
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #利用tile函数扩展预测数据，与每一个训练样本求距离

    sqDiffMat = diffMat**2 #对求出来的差分别平方
    sqDistances = sqDiffMat.sum(axis=1)#sum应该是默认的axis=0 就是普通的相加，而当加入axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5  #求欧式距离
    sortedDistIndicies = distances.argsort()#argsort函数返回的是数组值从小到大的索引值,记住返回值是数组的下标，可以这么理解
    classCount={}  #定义一个字典，用于储存K个最近点对应的分类以及出现的频次
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#字典里面的该标签数量++
    # 以下代码将不同labels的出现频次由大到小排列，输出次数最多的类别
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

