import numpy
import torch
import sys
from torch import nn
class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha  # [B,C,1,1]
            norm = self.gamma / \
                   (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            # [B,1,1,1],公式中的根号C在mean中体现
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                   (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        print((embedding * norm))
        print(self.beta)
        print(embedding * norm + self.beta)
        print("-----")

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        # 这里的1+tanh就相当于乘加操作
        print(x.size())
        print(gate.size())
        return x * gate


if __name__ == '__main__':
    gct = GCT(num_channels=3)
    x = [
        [
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            ],
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            ],
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            ]
        ],
        [
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            ],
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            ],
            [
                [1, 2, 3, 4],
                [1, 2, 3, 4]
            ]
        ]
    ]
    x = numpy.array(x)
    x = torch.FloatTensor(x)
    y = gct(x)
