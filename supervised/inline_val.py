# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：semanticSegmentation
@Date ：2022/2/22 18:47
"""
import torch
from data.voc_dataset2 import YQYvocDataSet_Feature, YQYvocDataSet_Feature_unlabel
import torch.nn as nn
import numpy as np
from util.loss import CrossEntropy2d, Gini
import network
from util import weights_init
import collections
from util.focusMap import getGini, saveGini

data_path = '/content/drive/My Drive/integrity_dataSet'
bin_path = '/content/drive/My Drive/integrity_dataSet/profile_list/'
checkpointPath = "./2_checkpoint/"
gini_path = './data/gini/'

# data_path = '/home/lyx/workspace/Disk412/python-workspace/dataSet/seismicData'
# bin_path = '/home/lyx/workspace/seismicFacies/CPS3/data/profile_list/'
# checkpointPath = "./DI_CPS/"
# gini_path = './data/gini/'

P = 8

total_epoch = 50
batch_size = 1
CLASS = 6  # 类别数
Validation_N = 82  # 测试集样本个数
Validation_batch_size = 2

if P == 2:
    N = 350
    save_batchs = [80, 160, 240, 349]
    labeledIndex_path = 'train_supervised1_2.txt'
    unlabeledIndex_path = 'train_unsupervised1_2.txt'
    validateIndex_path = 'validation1_2.txt'
if P == 4:
    N = 526
    save_batchs = [80, 160, 240, 320, 400, 480, 525]
    labeledIndex_path = 'train_supervised1_4.txt'
    unlabeledIndex_path = 'train_unsupervised1_4.txt'
    validateIndex_path = 'validation1_4.txt'
if P == 8:
    N = 612
    save_batchs = [80, 160, 240, 320, 400, 480, 560, 611]
    labeledIndex_path = 'train_supervised1_8.txt'
    unlabeledIndex_path = 'train_unsupervised1_8.txt'
    validateIndex_path = 'validation1_8.txt'
if P == 16:
    N = 656
    save_batchs = [80, 160, 240, 320, 400, 480, 560, 655]
    labeledIndex_path = 'train_supervised1_16.txt'
    unlabeledIndex_path = 'train_unsupervised1_16.txt'
    validateIndex_path = 'validation1_16.txt'


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

    aveJ, j_list = ConfM.jaccard()
    accuracy = ConfM.accuracy()
    recall = ConfM.recall()

    classes = np.array(('壹',  # always index 0
                        '贰', '叁', '肆', '伍',
                        '陆'))
    for key in j_list:
        print('class {}  IU {:.2f}'.format(key, j_list[key]))
    print('meanIOU: ' + str(aveJ) + '\n')

    return aveJ, j_list, accuracy, recall


# 模型加载
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_left = network.getNetwork().to(device)


# 初始化参数
weights_init.weights_init(model_left, "kaiming")


opt_l = torch.optim.Adam(model_left.parameters(), lr=0.001)
opt_l.zero_grad()

loss_func = CrossEntropy2d()
Gini = Gini()



# 加载测试集
test_dataset = YQYvocDataSet_Feature(data_path, bin_path + validateIndex_path)

iou_validation = []
iou_validation1 = []
iou_validation2 = []
loss_validation = []
validation_class_aveJ = dict()
validation_class_aveJ['one'] = []
validation_class_aveJ['two'] = []
validation_class_aveJ['three'] = []
validation_class_aveJ['four'] = []
validation_class_aveJ['five'] = []
validation_class_aveJ['six'] = []

acc_validation = []
acc_validation1 = []
acc_validation2 = []

recall_validation = []
recall_validation1 = []
recall_validation2 = []

aveJ_train = []
aveJ_train_dot = []
loss_train = []
loss_train_dot = []

pre_epoch = -1

# #加载模型：
PATH_l = checkpointPath + "checkpoint_9_epoch-611_l.pkl"
checkpoint_l = torch.load(PATH_l, map_location='cpu')
model_left.load_state_dict(checkpoint_l['model_state_dict'])
opt_l.load_state_dict(checkpoint_l['optimizer_state_dict'])


with torch.no_grad():
                # if batchs == 40 or batchs == 80 or batchs == 124:
                aveJ_train.append(np.mean(aveJ_train_dot))
                loss_train.append(np.mean(loss_train_dot))

                print("================验证=====================")
                iou_test_list = []
                acc_test_list = []
                recall_test_list = []
                loss_test_list = []
                iou_class_test_list = collections.defaultdict(list)
                for i in range(Validation_N // Validation_batch_size):
                    print('====测试集验证====', i, '/', Validation_N // Validation_batch_size,
                          '============================')
                    data = test_dataset.getData(i, Validation_batch_size)
                    data_shape = np.array(data['seismic'][0]).shape
                    validation_x0 = torch.FloatTensor(data['seismic']).to(device)
                    validation_x1 = torch.FloatTensor(data['pha']).to(device)
                    validation_y = torch.LongTensor(data['facies']).to(device)
                    print("样本：", data['names'])

                    interp = nn.Upsample(size=(data_shape[1], data_shape[2]), mode='bilinear',
                                         align_corners=True).float()
                    validation_out = interp(model_left(validation_x0, validation_x1))
                    _, max_test = torch.max(validation_out, dim=1)  # 第一个模型的输出结果

                    loss_test = loss_func(validation_out, validation_y)

                    # 计算IOU
                    data_list = []
                    out = max_test.detach().cpu().numpy()
                    train_y = np.array(validation_y.detach().cpu().numpy())
                    for i in range(out.shape[0]):
                        label = np.array(train_y[i])  # [462,587]
                        predict = np.array(out[i])  # [462,587]
                        data_list.append([label.flatten(), predict.flatten()])
                    aveJ_test, class_iou, acc, recall = get_iou(data_list, CLASS)
                    iou_test_list.append(aveJ_test)
                    acc_test_list.append(acc)
                    recall_test_list.append(recall)
                    for i in class_iou.keys():
                        iou_class_test_list[i].append(class_iou[i])
                    loss_test_list.append(loss_test.detach().cpu().numpy())
                    print('loss = {:.3f}'.format(loss_test))
                iou_validation.append(np.mean(iou_test_list))
                iou_validation1.append(np.mean(iou_test_list[10:30]))
                iou_validation2.append(np.mean(np.concatenate((iou_test_list[:10], iou_test_list[-10:]))))
                acc_validation.append(np.mean(acc_test_list))
                acc_validation1.append(np.mean(acc_test_list[10:30]))
                acc_validation2.append(np.mean(np.concatenate((acc_test_list[:10], acc_test_list[-10:]))))
                recall_validation.append(np.mean(recall_test_list))
                recall_validation1.append(np.mean(recall_test_list[10:30]))
                recall_validation2.append(np.mean(np.concatenate((recall_test_list[:10], recall_test_list[-10:]))))
                loss_validation.append(np.mean(loss_test_list))
                validation_class_aveJ['one'].append(np.mean(iou_class_test_list[0]))
                validation_class_aveJ['two'].append(np.mean(iou_class_test_list[1]))
                validation_class_aveJ['three'].append(np.mean(iou_class_test_list[2]))
                validation_class_aveJ['four'].append(np.mean(iou_class_test_list[3]))
                validation_class_aveJ['five'].append(np.mean(iou_class_test_list[4]))
                validation_class_aveJ['six'].append(np.mean(iou_class_test_list[5]))
                print("iou_validation结果：", iou_validation)
                print("iou_validation1结果：", iou_validation1)
                print("iou_validation2结果：", iou_validation2)
                print("loss_validation结果：", loss_validation)
                print("acc_validation结果：", acc_validation)
                print("acc_validation1结果：", acc_validation1)
                print("acc_validation2结果：", acc_validation2)
                print("recall_validation结果：", recall_validation)
                print("recall_validation1结果：", recall_validation1)
                print("recall_validation2结果：", recall_validation2)

