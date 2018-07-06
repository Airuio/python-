# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 17:10:45 2018

@author: a
"""
import numpy as np  
class perceptron(object):
    def __init__(self,eta = 0.1,n_iter = 10):
        self.eta = eta
        self .n_iter = n_iter
    def fit(self,x,y):
        self.w_ = np.zeros(1+x.shape[1])   #创建一个全为0的一维矩阵，长度为样本x特征数加1，加一是补充闵值的权重。
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(x,y):           #关联样本特征和预测值
                update = self.eta * (target - self.predict(xi) )
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0 )
            self.errors_.append(errors)
        return self     #//方法级联，自己返回修改后的自己
    def net_input(self,x):
        return np.dot(x, self.w_[1:])+self.w[0]         #计算预测值
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)     #三元语句，满足大于0返回1，不满足返回-1  
