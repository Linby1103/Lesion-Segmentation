## Lesion-Segmentation

## 1、开发环境

.python 3.6

.tensorflow1.12

.keras 2.15

. dicom 、SimpleITK

## 2、目录结构

.dataset 存放训练数据集生成代码和训练数据集

.model 存放网络构建代码

. ./存放训练代码和测试代码





## 3、项目流程—技术应用：

3.1 项目流程
数据预处理：
将医学影像（DICOM格式）进行转换，变为通用图像格式（PNG），以便用于深度学习模型训练。
数据增强(Data augmentation)：使用图像变换算法扩增数据集，提高模型的泛化能力。
训练图像分割模型（U-net）：
使用图像分割(segmentation)算法检测图像中所有可能是肺结节的区域，生成候选集。
训练三维卷积神经网络（3D-CNN）模型：
使用一种3D-CNN算法对上一步骤生成的结果进行分类，剔除假阳性的候选，保留真正的结节。
模型串联：
将两种模型进行串联，完整打通整个肺结节检测的流程，实现对输入的胸部CT图像进行肺结节检测。
3.2 技术应用
Python图像处理库的使用: opencv , scikit-image
Python医学影像处理库的使用: pydicom , SimpleITK
Python深度学习框架的使用: Keras, Tensorflow

## 4、CT图像

4.1 CT图像的格式 (DICOM)
DICOM（Digital Imaging and Communications in Medicine）即医学数字成像和通信，是医学图像和相关信息的国际标准（ISO 12052）。它定义了质量能满足临床需要的可用于数据交换的医学图像格式。

其它非标准格式：.mhd，.nii.gz

常见CT图像的规格是512*512像素，宽度和高度分别是512个像素点

## 5、图片分割模型

5.1 关键技术 – 基于CNN的分割网络

5.2 关键技术：U-net








