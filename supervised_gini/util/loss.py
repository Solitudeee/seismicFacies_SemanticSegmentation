import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        # self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # assert not target.requires_grad
        # assert predict.dim() == 4
        # assert target.dim() == 3
        # assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        # assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        # assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        # n, c, h, w = predict.size()

        # if not target.data.dim():
        #     return Variable(torch.zeros(1))

        #weight = [2.7161114,2.30869469,0.6446652,1.76149205,1.13636713,0.7008168,0.74062108,1.21182229,0.67425705,0.89285498]
        #weight = torch.FloatTensor(weight)

        #loss = F.cross_entropy(predict, target, weight=weight,size_average=self.size_average)
        loss = F.cross_entropy(predict, target,size_average=self.size_average)

        return loss



class Gini(nn.Module):

    def __init__(self):
        super(Gini, self).__init__()
        self.softmax = nn.Softmax()


    def forward(self, predict):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        batchsize, c, h, w = predict.size()
        predict = self.softmax(predict)
        print(batchsize,c,h,w,predict[0][0][0][0])
        gini = 1-torch.sum(torch.pow(predict,2),dim=1)
        gini = torch.div(torch.sum(gini),batchsize*h*w)
        print("gini的维度：",gini.size(),gini)

        return gini


# def Gini(predict):
#     #predict : batchsize,10,h,w
#     batchsize, c, h, w = predict.size()
#     gini = []
#
#     for i in range(batchsize):
#         x = predict[i]    #10,462,400
#         x = x.data.cpu().numpy()
#         x = 1-np.sum(np.power(x,2),axis=0)
#         # x = np.sum(np.power(x,2),axis=0)
#         x = np.array(x, dtype=np.float32)
#         # print("##############",np.max(x),np.min(x))
#         # x.tofile("a.bin")
#         gini.append(x)
#     gini = np.array(gini)
#     # print("gini的维度：",np.sum(gini))
#     print("gini的维度：",np.sum(gini)/(batchsize*462*400))
#     # print("gini的维度：",gini.shape)
#     return gini


class Tsallis(nn.Module):

    def __init__(self):
        super(Tsallis, self).__init__()


    def forward(self, predict):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        q = 0.6
        batchsize, c, h, w = predict.size()
        tsallis = 1-torch.sum(torch.pow(predict,q),dim=1)
        tsallis = torch.div(tsallis,(q-1))
        tsallis = torch.div(torch.sum(tsallis),batchsize*h*w)
        print("tsallis的维度：",tsallis.size(),tsallis)
        return tsallis


class Renyi(nn.Module):

    def __init__(self):
        super(Renyi, self).__init__()


    def forward(self, predict):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        q = 0.6
        batchsize, c, h, w = predict.size()
        renyi = torch.sum(torch.pow(predict,q),dim=1)
        renyi = torch.log(renyi)
        renyi = torch.div(renyi,(1-q))
        renyi = torch.div(torch.sum(renyi),batchsize*h*w)
        print("tsallis的维度：",renyi.size(),renyi)
        return renyi