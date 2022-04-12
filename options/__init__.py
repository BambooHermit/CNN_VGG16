#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Yvette time:2022/4/6


class BaseData:
    savepath = ['./data/x_train.npy', './data/y_train.npy', './data/x_test.npy','./data/y_test.npy']
    path = ['./datasets/train_data/', './datasets/test_data/']
    txt = ['./datasets/train_data.txt', './datasets/test_data.txt']
    checkpoint_save_path = "./checkpoint/VGG16.ckpt"

    def __init__(self, state, picSize):
        if state == 1:
            BaseData.picSize = picSize
        else:
            BaseData.picSize = 0


