# -*- coding: utf-8 -*-


import numpy as np

"""
參考網址: https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch02/ch02.py
"""


class Perceptron:
    """
    感知器(Perceptron這個演算法只有在資料是線性可分的形況下才能正確分類（演算法才會停止）)
    Perceptron Learning Algorithm，PLA
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        初始化
        :param eta: 學習率
        :param n_iter: 迭代處理的次數
        :param random_state: seed of random
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        :param X: array-like shape = [n_samples, n_feature]
                  training dataset， n_samples表示有幾筆，n_feature表示特徵
        :param y: array-like shape = [n_samples]
                  target values
        :return:
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print('[+] [Perceptron] fit，init weight arr shape: {}'.format(self.w_.shape))
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                # * xi
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            # 記錄每一個epochs的損失
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        計算含權重的加總合(權重第一個為偏差值)
        :param X: array-like shape = [n_samples, n_feature]
        :return:
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
