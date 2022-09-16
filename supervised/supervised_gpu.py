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

# data_path = '/media/lyx/412硬盘/python-workspace/dataSet/seismicData'
# bin_path = '/media/lyx/412硬盘/python-workspace/dataSet/seismicData/profile_list/'
# checkpointPath = "./1_checkpoint/"
# gini_path = './data/gini/'

data_path = '/content/drive/My Drive/integrity_dataSet'
bin_path = '/content/drive/My Drive/integrity_dataSet/profile_list/'
checkpointPath = "./1_checkpoint/"
gini_path = './data/gini/'


total_epoch = 15
batch_size = 2
CLASS = 6  # 类别数
Validation_N = 82  # 测试集样本个数
Validation_batch_size = 2

N = 700
save_batchs = [80, 160, 240, 349]
labeledIndex_path = 'train.txt'
validateIndex_path = 'validation.txt'


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


# 加载训练集
train_dataset_labeled = YQYvocDataSet_Feature(data_path, bin_path + labeledIndex_path, 111)
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

pre_epoch = 0

# #加载模型：
# PATH_l = checkpointPath + "checkpoint_0_epoch-349_l.pkl"
# checkpoint_l = torch.load(PATH_l, map_location='cpu')
# model_left.load_state_dict(checkpoint_l['model_state_dict'])
# opt_l.load_state_dict(checkpoint_l['optimizer_state_dict'])
# pre_epoch = checkpoint_l['epoch']
# iou_validation = checkpoint_l['iou_validation']
# iou_validation1 = checkpoint_l['iou_validation1']
# iou_validation2 = checkpoint_l['iou_validation2']
# loss_validation = checkpoint_l['loss_validation']
#
# acc_validation = checkpoint_l['acc_validation']
# acc_validation1 = checkpoint_l['acc_validation1']
# acc_validation2 = checkpoint_l['acc_validation2']
#
# recall_validation = checkpoint_l['recall_validation']
# recall_validation1 = checkpoint_l['recall_validation1']
# recall_validation2 = checkpoint_l['recall_validation2']
for epoch in range(pre_epoch + 1, total_epoch):
    for batchs in range(N // batch_size):
        print(epoch, '/', total_epoch, '--', batchs, '/', N // batch_size, '============================')

        # batchs:第batchs次训练，batch：大小
        labeled_data = train_dataset_labeled.getData(batchs, batch_size)

        names = labeled_data['names']


        data_shape = np.array(labeled_data['seismic'][0]).shape

        labeled_x0 = torch.FloatTensor(labeled_data['seismic']).to(device)
        labeled_x1 = torch.FloatTensor(labeled_data['pha']).to(device)
        # labeled_x2 = torch.FloatTensor(labeled_data['ifr2']).to(device)

        labeled_y = torch.LongTensor(labeled_data['facies']).to(device)


        interp = nn.Upsample(size=(data_shape[1], data_shape[2]), mode='bilinear', align_corners=True).float()


        # *******将数据输入网络******
        # if epoch > -1:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gini = torch.FloatTensor(getGini(gini_path, names, epoch - 1)).to(device)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        pred_l = interp(model_left(labeled_x0, labeled_x1, gini))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # else:
        #     pred_l = interp(model_left(labeled_x0, labeled_x1))
        #     if hasattr(torch.cuda, 'empty_cache'):
        #         torch.cuda.empty_cache()


        saveGini(pred_l, gini_path, names, epoch)

        _, max_l = torch.max(pred_l, dim=1)  # 第一个模型的输出结果


        max_l = max_l.long()

        # 有监督损失
        loss = loss_func(pred_l, labeled_y)  # 第一个模型的有监督损失

        # 更新参数w

        opt_l.zero_grad()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        loss.backward()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        opt_l.step()


        # 训练评价
        with torch.no_grad():
            if batchs % 10 == 0:
                data_list = []

                out = pred_l.detach().cpu().numpy()

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
                    if epoch==1 and batchs==80:
                        gini = torch.FloatTensor(getGini(gini_path, ['300.csv','303.csv'], 0)).to(device)
                        validation_out = interp(model_left(validation_x0, validation_x1, gini))
                    else:
                        gini = torch.FloatTensor(getGini(gini_path, data['names'], -1)).to(device)
                        validation_out = interp(model_left(validation_x0, validation_x1, gini))
                    _, max_test = torch.max(validation_out, dim=1)  # 第一个模型的输出结果

                    saveGini(validation_out, gini_path, data['names'], -1)

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
                with open("result.txt",'w') as f:
                    f.writelines("iou_validation结果：")
                    f.writelines(str(iou_validation))

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
