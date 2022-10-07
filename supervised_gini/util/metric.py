import os, sys
import numpy as np
from multiprocessing import Pool
import copyreg
import types


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.BlackList = []

    # def add(self, gt, pred):
    #     assert(np.max(pred) <= self.nclass)
    #     assert(len(gt) == len(pred))
    #     for i in range(len(gt)):
    #         if not gt[i] == 255:
    #             self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix['m'].shape == self.M.shape)
        self.M += matrix['m']
        self.BlackList.extend(matrix['blacklist'])

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        k = 0  # k个类在这个batch中都小于1%
        for i in range(self.nclass):
            if self.BlackList.count(i) == 2:
                k += 1
            else:
                xxx = np.sum(self.M[:, i])
                if xxx != 0:
                    recall += self.M[i, i] / xxx

        return recall / (self.nclass-k)

    def accuracy(self):
        accuracy = 0.0
        k = 0  # k个类在这个batch中都小于1%
        for i in range(self.nclass):
            if self.BlackList.count(i) == 2:
                k += 1
            else:
                accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / (self.nclass-k)

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:  # IoU的公式
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))
            else:
                jaccard_perclass.append(0)
        k = 0  # k个类在这个batch中都小于1%
        print("***********", self.BlackList)
        classIoU = {}
        for i in range(self.nclass):
            if self.BlackList.count(i) == 2:
                jaccard_perclass[i] = 0
                k += 1
            else:
                classIoU[i] = jaccard_perclass[i]

        return np.sum(jaccard_perclass) / (len(jaccard_perclass) - k), classIoU

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        # 类别占比小于1%，不参加精确度计算
        n = len(gt)
        gtList = list(gt.flatten())
        blacklist = []
        for i in range(self.nclass):
            if gtList.count(i) / n < 0.01:
                blacklist.append(i)
        assert (n == len(pred))
        for i in range(n):
            if gt[i] not in blacklist and gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        dict = {}
        dict['m'] = m
        dict['blacklist'] = blacklist
        return dict

# if __name__ == '__main__':
#     args = parse_args()
#     # args = get_arguments()
#
#     m_list = []
#     data_list = []
#     test_ids = [i.strip() for i in open(args.test_ids) if not i.strip() == '']
#     for index, img_id in enumerate(test_ids):
#         if index % 100 == 0:
#             print('%d processd'%(index))
#         pred_img_path = os.path.join(args.pred_dir, img_id+'.png')
#         gt_img_path = os.path.join(args.gt_dir, img_id+'.png')
#         pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
#         gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
#         # show_all(gt, pred)
#         data_list.append([gt.flatten(), pred.flatten()])
#
#     ConfM = ConfusionMatrix(args.class_num)
#     f = ConfM.generateM
#     pool = Pool()
#     m_list = pool.map(f, data_list)
#     pool.close()
#     pool.join()
#
#     for m in m_list:
#         ConfM.addM(m)
#
#     aveJ, j_list, M = ConfM.jaccard()
#     with open(args.save_path, 'w') as f:
#         f.write('meanIOU: ' + str(aveJ) + '\n')
#         f.write(str(j_list)+'\n')
#         f.write(str(M)+'\n')
