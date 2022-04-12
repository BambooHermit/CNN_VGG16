#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Yvette time:2022/4/7

import numpy as np
from data import create_dataset
from options import BaseData
from models import create_model
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from options import plot
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    data = BaseData(1, 32)

 #   x_train, y_train, x_test, y_test = create_dataset(data.savepath, data.path, data.txt, data.picSize)
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = create_model('cifar10')

    model.load_weights(data.checkpoint_save_path)

    predictions = model.predict(x_test)

    y_predict = np.argmax(predictions, 1)
    plot.plot_confusion_matrix(y_test, y_predict)
