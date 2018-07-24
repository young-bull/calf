import os
import numpy as np
import matplotlib.pyplot as plt

class Statistics(object):
        
    def __init__(self, yhat, y):
        self.yhat = yhat
        self.y = y
        self.err_detail = np.zeros(shape=(10,10)) # 纵轴：10类真实值，横轴：预测值
        self.err_idx = (yhat - y).nonzero()[0] # 两者相减，不为0即为不同，即判断出错
        
        for i in self.err_idx:
            self.err_detail[int(y[i]),int(yhat[i])] += 1
        
        self.FN = self.err_detail.sum(axis=1)
        self.TP = len(y)/10 - self.FN
        
        self.t = np.zeros(shape=(10,))
        for i in range(10):
            self.t[i] = (self.yhat==i).sum()
            
        self.FP = self.t - self.TP
          
    # 查全率，召回
    def Recall(self):
        return self.TP/(self.TP+self.FN)
        
    # 查准率，精度
    def Precision(self):
        return self.TP/(self.TP+self.FP)
                 
    def show(self):    
        bar_width_rate = 0.08
        plt.figure(figsize=(15,5)) 
        plt.bar(np.arange(10),self.err_detail.sum(axis=1),width=bar_width_rate*10)

        for j in range(10):
            plt.bar(np.arange(10)-bar_width_rate*5+bar_width_rate/2+j*bar_width_rate,
                    -self.err_detail.T[j],
                    label=j,
                    width=bar_width_rate,
                    color='C'+str(j))
        plt.xticks(np.arange(10))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()