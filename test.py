import kNN
import numpy as np
dataSet,labels = kNN.createDataSet()
input = np.array([0.0,0.1])
K = 3
output = kNN.classify0(input,dataSet,labels,K)
print("测试数据为:",input,"分类结果为：",output)