# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 18:18
# @Author  : XuWenzhong
# @Email   : xuwenzhong1994@163.com
# @File    : Day 3_Multiple_Linear_Regression.py
# @Version : 1.0.1
# @Software: PyCharm
# @Description:多元线性回归实现

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
'''
回归分析前提：
1、线性
2、保持误差项的方差齐性（常数方差）：误差项的分散（方差）必须等同
3、多元正太分布：多元回归假定残差符合正态分布
4、缺少多重共线性：假设数据有极少甚至没有多重共线性。当特征（或自变量）不是相互独立时，会引发多重共线性
'''

# 数据预处理，将类别数据数字化
dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

'''
多元回归模型中，数据集是非数值数据类型时，使用分类数据非常有效
分类数据：反映事物类别的数据，离散数据，数值个数有限但可能很多，且值之间无序，如性别
可采用虚拟变量替换，如0,1
'''
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# 躲避虚拟变量陷阱
'''
虚拟变量陷阱是指两个以上（包括两个）变量之间高度相关的情形。
简而言之，就是存在一个能够被其他变量预测出的变量
举一个存在重复类别（变量）的例子：
假设舍弃男性性别，那么该类别也可以通过女性类别来定义（女性值为0时，表示男性，为1，表示女性），反之亦然
解决办法：类别变量减一
假如有m个类别，那么在模型构建时取m-1个虚拟变量，减去的那个变量可以看做是参照值
'''
'''
注意：
    过多的变量可能会减低模型精确度，尤其是如果存在一些对结果无关的变量，或者存在对其他变量造成很大影响的变量。
    选择合适变量的方法：
    1、向前选择法
    2、向后选择法（也称 向后剔除法/向后消元法）
    3、向前向后法：先用向前法筛选一遍，再用向后法筛选一遍，直到最后无论怎么筛选模型变量都不在发生变化，就是结束了
'''
X = X[:, 1:]

# 拆分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 在训练集上训练多元线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 在测试集上预测结果
y_pred = regressor.predict(X_test)

print(r2_score(Y_test, y_pred))
