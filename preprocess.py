# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:30:15 2021

@author: 10437
"""
import numpy as np
import struct
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing  

 ##解析idx3文件的通用函数
def decode_idx3_ubyte(idx3_ubyte_file):
    #读取二进制文件
    bin_data = open(idx3_ubyte_file, 'rb').read()
    
    #解析文件信息
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    #解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    #print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    #print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows*num_cols,1))

    for i in range(num_images):
        #if (i + 1) % 10000 == 0:
            #print('已解析 %d' % (i + 1) + '张')
            #print(offset)
        min_max_scaler = preprocessing.MinMaxScaler() 
        images[i] =min_max_scaler.fit_transform(np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_cols*num_rows,1))) 
        offset += struct.calcsize(fmt_image)

    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    #print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        #if (i + 1) % 10000 == 0:
            #print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)

        
    return labels

def load_train_images(train_images_idx3_ubyte_file):
    idx_ubyte_file=train_images_idx3_ubyte_file
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(train_labels_idx1_ubyte_file):
    idx_ubyte_file=train_labels_idx1_ubyte_file
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(test_images_idx3_ubyte_file):
    idx_ubyte_file=test_images_idx3_ubyte_file
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(test_labels_idx1_ubyte_file):
    idx_ubyte_file=test_labels_idx1_ubyte_file
    return decode_idx1_ubyte(idx_ubyte_file)



