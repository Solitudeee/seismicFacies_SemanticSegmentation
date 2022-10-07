# -*- coding: UTF-8 -*-
"""
@author ：yqy
@Project ：semanticSegmentation01
@Date ：2021/9/30 11:26
"""

import torch
def SGD(params_list):
    return torch.optim.SGD(params_list,lr=base_lr,
                                  momentum=0.9,
                                  weight_decay=0.0005)