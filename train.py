#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Yvette time:2022/4/6

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

if __name__ == '__main__':
    data = BaseData(1, 32)
 #   x_train, y_train, x_test, y_test = create_dataset(data.savepath, data.path, data.txt, data.picSize)
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = create_model('cifar10')

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])



    model.load_networks(data.checkpoint_save_path)

    cp_callback = model.save_networks(data.checkpoint_save_path)

    history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                        callbacks=[cp_callback])
    model.summary()

    # 参数提取
    model.parameter_extraction()

    # 显示训练集和验证集的acc和loss曲线
    plot.plot_acc(history)
    plot.plot_loss(history)
