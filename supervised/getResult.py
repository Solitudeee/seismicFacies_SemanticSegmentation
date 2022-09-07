# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：semanticSegmentation
@Date ：2022/2/22 18:47
"""
import torch
import numpy as np
from util.loss import CrossEntropy2d
import network
from util import weights_init
import pandas as pd
import torch.nn as nn

data_path = '/home/lyx/workspace/dataSet/seismicFacies/seismic/'
label_path = '/home/lyx/workspace/dataSet/seismicFacies/label/'


checkpointPath = "./8_checkpoint/checkpoint_1_epoch-80_l.pkl"
resultPath = "./8_checkpoint/result/"



def get_iou(data_list, class_num):
    from multiprocessing import Pool
    from util.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)  # class_num:类别数
    f = ConfM.generateM

    pool = Pool(22)

    m_list = pool.map(f, data_list)  # 将数组中的每个元素提取出来当作函数的参数，创建一个个进程，放进进程池中
    # 第一个参数是函数，第二个参数是一个迭代器，将迭代器中的数字作为参数依次传入函数中
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)  # m(i,j)是一张图中预测为i真实为j的个数  M是一个batch中的m相加

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(('壹',  # always index 0
                        '贰', '叁', '肆', '伍',
                        '陆'))
    class_iou=[]
    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))
        class_iou.append(j_list[i])
    print('meanIOU: ' + str(aveJ) + '\n')

    return aveJ,class_iou

# 模型加载
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_left = network.getNetwork().to(device)


# 初始化参数
weights_init.weights_init(model_left, "kaiming")


opt_l = torch.optim.Adam(model_left.parameters(), lr=0.001)
opt_l.zero_grad()


loss_func = CrossEntropy2d()

# #加载模型：
PATH_l=checkpointPath
checkpoint_l = torch.load(PATH_l,map_location='cpu')
model_left.load_state_dict(checkpoint_l['model_state_dict'])
opt_l.load_state_dict(checkpoint_l['optimizer_state_dict'])

w = 590
h = 1006

# name = ["573.csv","1.csv"]  #1_2
# name = ["267.csv","2.csv"]  #1_4
name = ["303.csv","0.csv"]  #1_8
#读取样本===================================================
seismic = []
label = []
nums = []
for i in name:
    x = np.array(pd.read_csv(data_path+i, header=None))
    y = np.array(pd.read_csv(label_path+i, header=None))
    # 数据最大最小归一化
    smax = np.max(x)
    smin = np.min(x)
    x = (x - smin) / (smax - smin)
    x.shape = [1, w, h]
    y -= 1
    seismic.append(x)
    label.append(y)
    xx = np.unique(y.reshape((1, -1)))
    xx.sort()
    nums.append(xx)

n = len(name)
seismic = np.array(seismic)
seismic.shape = (n,1,w,h)
label = np.array(label)
label.shape = (n,w,h)

seismic = torch.tensor(seismic, dtype=torch.float32).to(device)
label = torch.LongTensor(label).to(device)
interp = nn.Upsample(size=(w, h), mode='bilinear', align_corners=True).float()
out = interp(model_left(seismic))

_, max = torch.max(out, dim=1)  # 第一个模型的输出结果

loss = loss_func(out, label)
# 计算IOU

out = max.detach().cpu().numpy()
label = np.array(label.detach().cpu().numpy())
print("nums:",nums)

for i in range(n):
    data_list = []
    oo = out[i] + 1
    pd.DataFrame(oo).to_csv(resultPath+name[i], header=False, index=False)
    y = np.array(label[i]).flatten()
    predict = np.array(out[i]).flatten()
    data_list.append([y,predict])
    print(nums[i])
    aveJ_test, class_iou = get_iou(data_list, 6)



