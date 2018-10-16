# -*- coding: utf-8 -*-
# @Time    : 2018/10/15 19:13
# @Author  : XuWenzhong
# @Email   : xuwenzhong1994@163.com
# @File    : Day 1_Data_Preprocessing.py
# @Version : 1.0.1
# @Software: PyCharm
# @Description:数据预处理

import numpy as np     # 包含数学计算函数
import pandas as pd    # 导入和管理数据集
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 导入数据集
'''
读取CSV文件为一个数据帧，从中制作自变量和因变量的矩阵和向量
'''
dataset = pd.read_csv('../datasets/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
print("Step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

# 处理丢失数据
'''
处理因为各种原因丢失的数据，以不降低机器学习模型的性能
可以用整列数据的平均值或中间值替换丢失数据
使用Imputer类
'''
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)

# 解析分类数据
'''
分类数据指的是含有标签值而不是数字值的变量
取值范围通常是固定的
例如"Yes"和"No"不能用于模型的数学计算，所以需要解析成数字
使用LabelEncoder类
'''
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# 创建虚拟变量
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

# 拆分数据集为训练集合和测试集合
'''
训练集合用来训练模型，测试集合用来验证模型 两者一般比例是80:20
'''
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

# 特征量化
'''
大部分模型算法使用两点间的欧式距离表示，但此特征在幅度、单位和范围姿态问题上变化很大
在距离计算中，高幅度的特征比低幅度的特征权重更大，可用特征标准化或Z值归一化解决
使用StandardScaler类
'''
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)

