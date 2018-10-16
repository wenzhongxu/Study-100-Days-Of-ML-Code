# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 17:01
# @Author  : XuWenzhong
# @Email   : xuwenzhong1994@163.com
# @File    : Day 2_Simple_Linear_Regression.py
# @Version : 1.0.1
# @Software: PyCharm
# @Description:简单线性回归实现

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 数据预处理
dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4, random_state=0)

# 训练集使用简单线性回归模型来训练
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 预测结果
Y_pred = regressor.predict(X_test)

# 训练集结果可视化
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

# 测试集结果可视化
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()
