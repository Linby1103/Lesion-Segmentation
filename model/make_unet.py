# @File  : make_unet.py
# @Author: LiBin
# @Date  : 2020/3/20
# @Desc  :
import csv
import glob
import random
import cv2
import numpy
import os
from typing import List, Tuple
from keras.optimizers import SGD
from keras.layers import *
from keras.models import *
"""
构建Unet网络
"""
class Build_Unet(object):
    def __init__(self, input_shape):
        self.shape=input_shape

    """"
    @func:Unet 网络构建
    @param:None
    @return: 构建的网络输出
    """

    def make_model(self):
        inputs = Input(shape=self.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print(conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print(conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print(pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print(conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print(conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print(pool2.shape)
        print('\n')

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print(conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print(conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print(pool3.shape)
        print('\n')

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        print(conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print(conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print(pool4.shape)
        print('\n')

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        print(conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        print(conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print('\n')

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        print(up6.shape)
        print(drop4.shape)
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        print('merge: ')
        print(merge6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # conv10 = Softmax()(conv9)
        print(conv10.shape)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=SGD(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        print('model compile')
        return model



