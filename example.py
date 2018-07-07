# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:08:57 2018

@author: a
"""

from perceptron_achieve import perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
df = pd.read_excel(io = 'lris.xlsx',header = None)    #读取数据为Dataframe结构，没有表头行
y = df.iloc[0:100,4].values         #取前100列数据，4列为标识
y = np.where(y == 'Iris-setosa', -1,1)
X = df.iloc[0:100,[0,2]].values  #iloc为选取表格区域，此处取二维特征进行分类,values为返回不含索引的表

plt.scatter(X[:50,0],X[0:50,1],color = 'red',marker = 'o', label = 'setosa')
plt.scatter(X[50:100,0],X[50:100,1],color = 'blue',marker = 'x', label = 'versicolor')
plt.xlabel('petal lenth')
plt.ylabel('sepal lenth')
plt.legend(loc = 2)    #画出标签以及标签的位置参数      
plt.show()             #出图
#以上六行与分类无关，仅仅是为了直观的感受两块数据的分布区域
ppn = perceptron(eta = 0.1,n_iter = 10)
ppn.fit(X,y)  #完成了分类的任务，下面进行结果展示

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker = 'o')
plt.xlabel('Epoches')
plt.ylabel('numbers of wrong classfications')
plt.xlim((1,10))
plt.ylim((0,4))
plt.show()    #该图展示了迭代次数的收敛结果至第六次完全收敛

print(ppn.w_)  #打印结果，下面对分类结果做可视化处理

def plot_decision_region(X,y,classifier,resolution = 0.02):
    markers = ('s','x','o','~','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #画出界面
    x1_min, x1max = X[:,0].min() - 1, X[:,0].max() + 1   
    x2_min, x2max = X[:,1].min() - 1, X[:,1].max() + 1  
    
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1max,resolution),  
                          np.arange(x2_min,x2max,resolution))   #生成均匀网格点，
    '''meshgrid的作用是根据传入的两个一维数组参数生成两个数组元素的列表。如果第一个参数是xarray，
    维度是xdimesion，第二个参数是yarray，维度是ydimesion。那么生成的第一个二维数组是以xarray为行，
    ydimesion行的向量；而第二个二维数组是以yarray的转置为列，xdimesion列的向量。'''
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    #在全图上每一个点（间隔0.2）计算预测值，并返回1或-1
    
    plt.contourf(xx1,xx2,Z,alpha = 0.5,cmap = cmap) #画出等高线并填充颜色
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #画上分类后的样本
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],alpha=0.8,
                    c=cmap(idx),marker=markers[idx],label=cl)

plot_decision_region(X,y,classifier = ppn)
plt.xlabel('sepal lenth [cm]')
plt.ylabel('petal lenth [cm]')    
plt.legend(loc = 2)
plt.show()
    
    
    

