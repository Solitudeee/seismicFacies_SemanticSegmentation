# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：地震相实验
@Date ：2022/1/12 11:37
"""
import numpy as np


def removeSamples(A, B):
    A = list(A)
    for sample in B:
        A.remove(sample)
    return A


# with open("allData.txt",'w') as f:
#     for i in range(782):
#         f.writelines("%s_%s.csv\n"%(i,1))
#         f.writelines("%s_%s.csv\n"%(i,2))


ilines = np.arange(0, 782)

np.random.seed(111)
# 验证集样本
validation = np.random.choice(ilines, 82, replace=False)
validation.sort()
# 训练集样本
train = removeSamples(ilines, validation)

# 验证集文件
with open("validation.txt", 'w') as f:
    for i in validation:
        f.writelines("%s.csv\n" % (i))

# 训练集
with open("train.txt", 'w') as f:
    for i in train:
        f.writelines("%s.csv\n" % (i))


# 1_2的
train_supervised1_2 = ilines[215:565]  # 350
np.random.seed(222)

train_unsupervised1_2 = removeSamples(ilines, train_supervised1_2)
validation1_2 = np.random.choice(train_unsupervised1_2, 82, replace=False)
validation1_2.sort()
train_unsupervised1_2 = removeSamples(train_unsupervised1_2, validation1_2)

with open("train_supervised1_2.txt", 'w') as f:
    for i in train_supervised1_2:
        f.writelines("%s.csv\n" % (i))

with open("train_unsupervised1_2.txt", 'w') as f:
    for i in train_unsupervised1_2:
        f.writelines("%s.csv\n" % (i))

with open("validation1_2.txt", 'w') as f:
    for i in validation1_2:
        f.writelines("%s.csv\n" % (i))


# 1_4的
train_supervised1_4 = ilines[303:478]  # 175
np.random.seed(444)

train_unsupervised1_4 = removeSamples(ilines, train_supervised1_4)
validation1_4 = np.random.choice(train_unsupervised1_4, 82, replace=False)
validation1_4.sort()
train_unsupervised1_4 = removeSamples(train_unsupervised1_4, validation1_4)

with open("train_supervised1_4.txt", 'w') as f:
    for i in train_supervised1_4:
        f.writelines("%s.csv\n" % (i))

with open("train_unsupervised1_4.txt", 'w') as f:
    for i in train_unsupervised1_4:
        f.writelines("%s.csv\n" % (i))

with open("validation1_4.txt", 'w') as f:
    for i in validation1_4:
        f.writelines("%s.csv\n" % (i))

#1_8的
train_supervised1_8 = ilines[346:434]   #88
np.random.seed(888)

train_unsupervised1_8 = removeSamples(ilines,train_supervised1_8)
validation1_8 = np.random.choice(train_unsupervised1_8,82,replace=False)
validation1_8.sort()
train_unsupervised1_8 = removeSamples(train_unsupervised1_8,validation1_8)

with open("train_supervised1_8.txt", 'w') as f:
    for i in train_supervised1_8:
        f.writelines("%s.csv\n" % (i))

with open("train_unsupervised1_8.txt", 'w') as f:
    for i in train_unsupervised1_8:
        f.writelines("%s.csv\n" % (i))

with open("validation1_8.txt", 'w') as f:
    for i in validation1_8:
        f.writelines("%s.csv\n" % (i))

# 1_16的
train_supervised1_16 = ilines[369:412]  # 43
np.random.seed(6666)

train_unsupervised1_16 = removeSamples(ilines, train_supervised1_16)
validation1_16 = np.random.choice(train_unsupervised1_16, 82, replace=False)
validation1_16.sort()
train_unsupervised1_16 = removeSamples(train_unsupervised1_16, validation1_16)

with open("train_supervised1_16.txt", 'w') as f:
    for i in train_supervised1_16:
        f.writelines("%s.csv\n" % (i))

with open("train_unsupervised1_16.txt", 'w') as f:
    for i in train_unsupervised1_16:
        f.writelines("%s.csv\n" % (i))

with open("validation1_16.txt", 'w') as f:
    for i in validation1_16:
        f.writelines("%s.csv\n" % (i))




