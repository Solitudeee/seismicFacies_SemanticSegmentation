# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：semanticSegmentation
@Date ：2021/9/23 17:09
"""

import torch.nn as nn
import torch.nn.init as init
def weights_init(model,method):
    if method == "xavier":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)

                init.xavier_normal(m.weight)
                # xavier均匀分布的方法来init，来自2010年的论文“Understanding the difficulty of training deep feedforward neural networks”
                # init.kaiming_normal(m.weight)
                # 来自2015年何凯明的论文“Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()
    elif method == "kaiming":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):

                init.kaiming_normal(m.weight)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_()