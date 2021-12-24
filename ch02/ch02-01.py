# -*- coding: utf-8 -*-

"""
測試分類器效果
Reading-in the Iris data

https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(".")
import perceptron


# load data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print('[+] df data ')
print(df.tail())
print('[+] df End ')

# 取前100個，target
y = df.iloc[0:100, 4].values

# 轉換標籤
y = np.where(y == 'Iris-setosa', -1, 1)
print('[+] y data shape: {} '.format(y.shape))

# 取前100個，0欄位-花顎、2欄位-花瓣，來進行辨別；(先使用2維測試)
X = df.iloc[0:100, [0, 2]].values
print('[+] X data shape: {} '.format(X.shape))

# 看一下資料X(前50個都是Iris-setosa、後50個Iris-versicolor)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()



