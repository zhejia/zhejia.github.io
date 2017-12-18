# import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
# '''
# 1. numpy.c_ = <numpy.lib.index_tricks.CClass object>
#
# 将切片对象沿第二个轴（按列）转换为连接。
#
# 例子：
#
# [python] view plain copy
#
#     np.c_[np.array([1,2,3]), np.array([4,5,6])]
#     Out[96]:
#     array([[1, 4],
#            [2, 5],
#            [3, 6]])
#
#     np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
#     Out[97]: array([[1, 2, 3, 0, 0, 4, 5, 6]])
#
#
# 2. numpy.squeeze(a)
#
# 从数组的形状中删除单维条目，即把shape中为1的维度去掉
#
# [python] view plain copy
#
#     x = np.array([[[0], [1], [2]]])
#
#     x.shape
#     Out[99]: (1, 3, 1)
#
#     np.squeeze(x).shape
#     Out[100]: (3,)
# '''
# # print(np.array([1,2,3]))
# # print(np.array([1,2,3]).shape)
# # print(np.c_[np.array([1,2,3]), np.array([4,5,6])])
# # print(np.array([[1,2,3]]))
# # print(np.array([[1,2,3]]).shape)
# # print(np.c_[np.array([[1,2,3],[7,8,9]]), np.array([[4,5,6],[10,11,12]])])
"""


class linear_regression():
    def fit(self,train_X,train_Y,learning_rate=0.01,lamda=0.03,regularization="l2"):
        '''
        Args:
            train_X: 维度(n,m)，n 是特征维度，m是样本数
            train_Y: 维度(1,m)，m 是样本数
            learning_rate:
            lamda:
            regularization:

        Returns:
        '''


        feature_cnt,m = np.shape(train_X)
        self.W = np.random.randn(1,feature_cnt)
        self.b = np.zeros([1,1])

        step = 0
        past_best_error = sys.maxsize
        past_step = 0
        while step < sys.maxsize:
            pred = np.dot(self.W,train_X)+self.b

            self.W -= learning_rate*(lamda/m*self.W + np.dot((pred-train_Y),train_X.T))
            self.b -= learning_rate*np.sum(pred-train_Y)

            J_theta = np.sum((pred - train_Y)**2)/(2.0*m)

            if J_theta < past_best_error - 1e-6:
                past_best_error = J_theta
                past_step = step
            elif step - past_step > 10:
                break
            if step % 50 == 0:
                print("step %s: %s" %(step, J_theta))
            step += 1

    def predict(self,X):
        return np.dot(self.W,X)+self.b

    def score(self):
        '''
        Returns:R^2系数
        '''
        pass

def feature_label_split(pd_data):
    row_cnt = pd_data.shape[0]
    column_cnt = len(pd_data.iloc[0,0].split(';'))
    X = np.empty((row_cnt, column_cnt-1))
    Y = np.empty((row_cnt, 1))
    for i in range(0, row_cnt):
         row_array = pd_data.iloc[i,0].split(';')
         X[i] = np.array(row_array[0:-1])
         Y[i] = np.array(row_array[-1])
    return X, Y

def uniform_norm(X):
    """把特征标准化为均匀分布"""
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    return (X - X_min) / (X_max - X_min), X_max, X_min

if __name__ == "__main__":

    train_data = pd.read_csv("./wine/train.csv")
    test_data = pd.read_csv("./wine/test.csv")

    train_X, train_Y = feature_label_split(train_data)
    test_X, test_Y = feature_label_split(test_data)

    unif_train_X, max_X, min_X = uniform_norm(train_X)
    unif_test_X = (test_X - min_X )/( max_X - min_X)

    unif_train_X = unif_train_X.T   # X shape : (3414, 11)
    unif_test_X = unif_test_X.T
    train_Y = train_Y.T
    test_Y = test_Y.T

    model = linear_regression()
    model.fit(unif_train_X,train_Y,learning_rate=0.0001)

    print("训练集上效果评估 >>")
    pred_train = model.predict(unif_train_X)
    # print("R^2系数 ", model.score(unif_train_X, train_Y))
    print("均方误差 ",mean_squared_error(train_Y, pred_train))

    print("\n测试集上效果评估 >>")
    # r2 = model.score(unif_test_X, test_Y)
    # print("R^2系数 ",r2)
    pred = model.predict(unif_test_X)
    print("均方误差 ",mean_squared_error(test_Y, pred))
    # 等价于 print sum((pred - test_Y) ** 2)/test_Y.shape[0]

    # 下面对测试集上的标注值与预测值进行可视化呈现

    t = np.arange(len(pred.T[:200]))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, test_Y.T[:200], 'r-', lw=2, label=u'true value')
    plt.plot(t, pred.T[:200], 'b-', lw=2, label=u'estimated')
    plt.legend(loc = 'best')
    plt.title(u'Boston house price', fontsize=18)
    plt.xlabel(u'case id', fontsize=15)
    plt.ylabel(u'house price', fontsize=15)
    plt.grid()
    plt.show()

