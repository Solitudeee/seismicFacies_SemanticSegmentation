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
checkpointPath = "./8_checkpoint/"
gini_path = './data/gini/'

# data_path = '/media/lyx/412硬盘/python-workspace/dataSet/seismicData'
# bin_path = '/home/lyx/workspace/seismicFacies/CPS3/data/profile_list/'
# checkpointPath = "./8_checkpoint/"
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
# device = torch.device("cpu")
model_left = network.getNetwork().to(device)
model_right = network.getNetwork().to(device)

# 初始化参数
weights_init.weights_init(model_left, "kaiming")
weights_init.weights_init(model_right, "kaiming")

opt_l = torch.optim.Adam(model_left.parameters(), lr=0.001)
opt_l.zero_grad()

opt_r = torch.optim.Adam(model_right.parameters(), lr=0.001)
opt_r.zero_grad()

loss_func = CrossEntropy2d()
# Gini = Gini()

# 加载训练集
train_dataset_labeled = YQYvocDataSet_Feature(data_path, bin_path + labeledIndex_path, 111)

train_dataset_unlabeled = YQYvocDataSet_Feature_unlabel(data_path, bin_path + unlabeledIndex_path, 222)

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
PATH_l = checkpointPath + "checkpoint_5_epoch-611_l.pkl"
checkpoint_l = torch.load(PATH_l, map_location='cpu')
model_left.load_state_dict(checkpoint_l['model_state_dict'])
opt_l.load_state_dict(checkpoint_l['optimizer_state_dict'])
pre_epoch = checkpoint_l['epoch']
iou_validation = checkpoint_l['iou_validation']
iou_validation1 = checkpoint_l['iou_validation1']
iou_validation2 = checkpoint_l['iou_validation2']
loss_validation = checkpoint_l['loss_validation']

acc_validation = checkpoint_l['acc_validation']
acc_validation1 = checkpoint_l['acc_validation1']
acc_validation2 = checkpoint_l['acc_validation2']

recall_validation = checkpoint_l['recall_validation']
recall_validation1 = checkpoint_l['recall_validation1']
recall_validation2 = checkpoint_l['recall_validation2']

PATH_r = checkpointPath + "checkpoint_0_epoch-611_r.pkl"
checkpoint_r = torch.load(PATH_r, map_location='cpu')
model_right.load_state_dict(checkpoint_r['model_state_dict'])
opt_r.load_state_dict(checkpoint_r['optimizer_state_dict'])

for epoch in range(pre_epoch + 1, total_epoch):
    for batchs in range(N // batch_size):
        print(epoch, '/', total_epoch, '--', batchs, '/', N // batch_size, '============================')

        # batchs:第batchs次训练，batch：大小
        labeled_data = train_dataset_labeled.getData(batchs, batch_size)
        unlabeled_data = train_dataset_unlabeled.getData(batchs, batch_size)
        names = labeled_data['names']
        names.extend(unlabeled_data['names'])

        data_shape = np.array(labeled_data['seismic'][0]).shape

        labeled_x0 = torch.FloatTensor(labeled_data['seismic']).to(device)
        labeled_x1 = torch.FloatTensor(labeled_data['pha']).to(device)
        labeled_x2 = torch.FloatTensor(labeled_data['ifr2']).to(device)

        labeled_y = torch.LongTensor(labeled_data['facies']).to(device)

        unlabeled_x0 = torch.FloatTensor(unlabeled_data['seismic']).to(device)
        unlabeled_x1 = torch.FloatTensor(unlabeled_data['pha']).to(device)
        unlabeled_x2 = torch.FloatTensor(unlabeled_data['ifr2']).to(device)

        interp = nn.Upsample(size=(data_shape[1], data_shape[2]), mode='bilinear', align_corners=True).float()

        x0 = torch.cat([labeled_x0, unlabeled_x0], dim=0)
        x1 = torch.cat([labeled_x1, unlabeled_x1], dim=0)
        x2 = torch.cat([labeled_x2, unlabeled_x2], dim=0)

        # *******将数据输入网络******
        if epoch > 1:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gini = torch.FloatTensor(getGini(gini_path, names, 0)).to(device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            pred_l = interp(model_left(x0, x1, gini))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            pred_r = interp(model_right(x0, x2, gini))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:

            pred_l = interp(model_left(x0, x1))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            pred_r = interp(model_right(x0, x2))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        saveGini(pred_l, gini_path, names, 0)

        _, max_l = torch.max(pred_l, dim=1)  # 第一个模型的输出结果
        _, max_r = torch.max(pred_r, dim=1)  # 第二个模型的输出结果

        max_l = max_l.long()
        max_r = max_r.long()

        # 交叉损失
        cps_loss = loss_func(pred_l, max_r) + loss_func(pred_r, max_l)

        # 有监督损失
        loss_sup_l = loss_func(pred_l[:1], labeled_y)  # 第一个模型的有监督损失

        loss_sup_r = loss_func(pred_r[:1], labeled_y)  # 第二个模型的有监督损失

        loss = 0.5 * cps_loss + loss_sup_r + loss_sup_l

        # 更新参数w

        opt_l.zero_grad()
        opt_r.zero_grad()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        loss.backward()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        opt_l.step()
        opt_r.step()

        # 训练评价
        with torch.no_grad():
            if batchs % 10 == 0:
                data_list = []

                out = pred_l[:1].detach().cpu().numpy()

                train_y = np.array(labeled_y.detach().cpu().numpy())
                for i in range(out.shape[0]):
                    label = np.array(train_y[i])  # [462,587]
                    predict = np.array(out[i])  # [10,462,587]
                    predict = predict.transpose(1, 2, 0)
                    predict = np.asarray(np.argmax(predict, axis=2), dtype=np.int)
                    data_list.append([label.flatten(), predict.flatten()])
                aveJ, _,_,_ = get_iou(data_list, CLASS)
                aveJ_train_dot.append(aveJ)
            loss_train_dot.append(loss.detach().cpu().numpy())
        with torch.no_grad():
            if batchs in save_batchs:
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

                # 保存模型
                PATH_L = checkpointPath + "checkpoint_{}_epoch-{}_l.pkl".format(epoch, batchs)
                torch.save({
                    'epoch': epoch,
                    'batchs': batchs,
                    'model_state_dict': model_left.state_dict(),
                    'optimizer_state_dict': opt_l.state_dict(),
                    "iou_validation": iou_validation,
                    "iou_validation1": iou_validation1,
                    "iou_validation2": iou_validation2,

                    "acc_validation": acc_validation,
                    "acc_validation1": acc_validation1,
                    "acc_validation2": acc_validation2,

                    "recall_validation": recall_validation,
                    "recall_validation1": recall_validation1,
                    "recall_validation2": recall_validation2,

                    "loss_validation": loss_validation,
                    'aveJ_train': aveJ_train,
                    'loss_train': loss_train,
                    "validation_class_aveJ": validation_class_aveJ,
                }, PATH_L)
                PATH_R = checkpointPath + "checkpoint_{}_epoch-{}_r.pkl".format(epoch, batchs)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_right.state_dict(),
                    'optimizer_state_dict': opt_r.state_dict(),
                }, PATH_R)
