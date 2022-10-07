# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：地震相实验
@Date ：2022/3/20 13:48
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd


class Gini(nn.Module):

    def __init__(self):
        super(Gini, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, predict):
        # batchsize 样本个数   C 类别数
        batchsize, C = predict.size()
        predict = self.softmax(predict)
        gini = []  # 记录每个样本的Gini
        for i in range(batchsize):
            gini_one = 1 - torch.sum(torch.pow(predict[i], 2))
            gini.append(gini_one)
        gini = torch.tensor(gini)
        gini = torch.mean(gini.float())

        return gini


softmax = nn.Softmax()


def saveGini(predict,path,names,epoch):
    # predict : batchsize,10,h,w
    for i in range(len(names)):
        x = softmax(predict[i]).data.cpu().numpy()
        x = 1 - np.sum(np.power(x, 2), axis=0)
        # x = np.sum(np.power(x,2),axis=0)

        smax = np.max(x)
        smin = np.min(x)
        x = (x - smin) / (smax - smin)

        x = np.array(x, dtype=np.float32)
        pd.DataFrame(x).to_csv(path+"{0}".format(epoch)+'/'+names[i], header=False, index=False)


def getGini(path, names,epoch):
    gini = []
    for name in names:
        x = np.array(pd.read_csv(path + "{0}".format(epoch)+'/' + name, header=None))
        w, h = x.shape
        x.shape = (1, w, h)
        gini.append(x)
    return gini
