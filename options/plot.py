#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Yvette time:2022/4/7

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_acc(history):
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_prediction, labels = range(10), title = "Confusion matrix", save = False, save_path = None, cmap = plt.cm.Blues):
    cfm = confusion_matrix(y_true, y_prediction)

    cfm = np.around(cfm.astype('float')/cfm.sum(axis=1)[:, np.newaxis], decimals=2)
    figure = plt.figure(figsize=(10, 10))                   # 建立一个显示画面
    plt.imshow(cfm, interpolation='nearest', cmap=cmap)     # 根据cmp的数值大小在画面中填入颜色
    plt.title(title)                                        # 添加标题
    tick_index = np.arange(len(labels))                     # 刻度
    plt.yticks(tick_index, labels)                          # y轴类别名
    plt.xticks(tick_index, labels)                          # x轴类别名
    plt.colorbar()                                          # 生成颜色刻度

    threshold = cfm.max() / 2.                              # 在每一格Confusion matrix输入预测百分比

    for i in range(cfm.shape[0]):
        for j in range(cfm.shape[1]):
            color = "white" if cfm[i, j] > threshold else "black"
            # 如果格内背景颜色太深，则使用白色文字展示，反之使用黑色文字
            plt.text(j, i, cfm[i, j], horizontalalignment="center", color=color)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()                              # 将图片的位置进行调整，避免x或y轴文字被遮盖

    if save:
        plt.savefig("./confusion_matrix.png")

    plt.show()
