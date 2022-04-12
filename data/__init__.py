#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Yvette time:2022/4/6

from PIL import Image
import numpy as np
import os

# 自制数据集的构建
# def generateds(图片路径,标签文件)
def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readline()
    f.close()

    x, y_ = [], []

    for content in contents:
        value = content.split()
        img_path = value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))    # 具体功能和修改方式未知
        img = img / 255.
        x.append(img)
        y_.append(value[1])
        print('loading: ' + content)

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_

def create_dataset(savepath, path, txt, picSize):
    x_train_savepath = savepath[0]
    x_test_savepath = savepath[2]
    y_train_savepath = savepath[1]
    y_test_savepath = savepath[3]
    if os.path.exists(x_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(
            y_train_savepath) and os.path.exists(y_test_savepath):
        print('------------------ Load Datasets ---------------------')
        x_train_save = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test_save = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)

        x_train = np.reshape(x_train_save, (len(x_train_save), picSize, picSize))
        x_test = np.reshape(x_test_save, (len(x_test_save), picSize, picSize))

    else:
        print('------------------ Generate Datasets -----------------')
        x_train, y_train = generateds(path[0], txt[0])
        x_test, y_test = generateds(path[1], txt[1])

        print('------------------ Save Datasets ---------------------')
        x_train_save = np.reshape(x_train, (len(x_train), -1))
        x_test_save = np.reshape(x_test, (len(x_test), -1))
        np.save(x_train_savepath, x_train_save)
        np.save(x_test_savepath, x_test_save)
        np.save(y_train_savepath, y_train)
        np.save(y_test_savepath, y_test)

    return x_train, y_train, x_test, y_test












