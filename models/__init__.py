#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Yvette time:2022/4/6

import tensorflow as tf
import os
from models.BaseModel import VGG16


def create_model(name):
    model = VGG16()
    print("model [%s] was created" % name)
    return model
