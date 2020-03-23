# @File  : Dicom2JPEG.py
# @Author: LiBin
# @Date  : 2020/3/18
# @Desc  :
"""
读取dicom图像并将其转换为png图像
读取某文件夹内的所有dicom文件
:param src_dir: dicom文件夹路径
:return: dicom list
"""
import Augmentor
import os
import SimpleITK
import dicom
import numpy as np
import cv2
from tqdm import tqdm

from scipy import ndimage as ndi
import xml.etree.ElementTree as ET

##########################
def ReadCoorFromXml(path):
    global defectsum,total
    clsname = []
    in_file = open(path, 'r', encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    coordinate=[]
    objects = root.findall('object')

    for object in objects:

        bbox = object.find('bndbox')
        xmin = bbox.find('xmin').text.strip()
        xmax = bbox.find('xmax').text.strip()
        ymin = bbox.find('ymin').text.strip()
        ymax = bbox.find('ymax').text.strip()

        coordinate.append((int(xmin),int(ymin),int(xmax),int(ymax)))
    in_file.close()
    return coordinate

#########################


def is_dicom_file(filename):
    # 判断某文件是否是dicom格式的文件

    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False


def load_patient(src_dir):
    # 读取某文件夹内的所有dicom文件
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = dicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] \
                                 - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    # 读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array




'''
将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
:param image 输入的图像数组
:return: 归一化处理后的图像数组
'''
def normalize_hu(image):
    # 将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def ImageWrite(jpeg_data,image_name):
    if image_name != '':
        cv2.imwrite(jpeg_data,image_name)


def listdir(base_dir):
    data_txt='./path.txt'#dicom文件的路径
    with open(data_txt,'w') as fp:
        father_dir = os.listdir(base_dir)
        for dir in father_dir:  # 第一层
            child_dir = os.path.join(base_dir, dir)
            child_dir_next = os.listdir(child_dir)
            for sub_dir in child_dir_next:  # 第二层
                dicom_dir=child_dir+'/'+sub_dir
                fp.write(dicom_dir+'\n')

'''
数据增强
:param image 输入的图像数组
:return: 归一化处理后的图像数组
'''
ORIGIN_MERGED_SOURCE_DIRECTORY=''
AUGMENT_OUTPUT_DIRECTORY=''
TRAIN_SET_SIZE=''
VALIDATION_SET_SIZE=''
TEST_SET_SIZE=''
def augment():
	p = Augmentor.Pipeline(
		source_directory=ORIGIN_MERGED_SOURCE_DIRECTORY,
		output_directory=AUGMENT_OUTPUT_DIRECTORY
	)
	p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)

	p.sample(n=TRAIN_SET_SIZE + VALIDATION_SET_SIZE + TEST_SET_SIZE)

"""
Unet 分割数据集准备
"""
# 准备U-net训练数据



MASK_MARGIN = 5

def make_mask(v_center, v_diam, width, height):
    mask = np.zeros([height, width])
    v_xmin = np.max([0, int(v_center[0] - v_diam) - MASK_MARGIN])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + MASK_MARGIN])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - MASK_MARGIN])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + MASK_MARGIN])
    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = v_x
            p_y = v_y
            if np.linalg.norm(np.array([v_center[0], v_center[1]])\
                                 - np.array([p_x, p_y]))<= v_diam * 2:
                mask[p_y, p_x] = 1.0   #设置节点区域的像素值为1
    return (mask)

import glob
def Test():
    tsave_path='D:/install/Unet/Lung_detect/dataset/train_dataset/'
    lsave_path='D:/install/Unet/Lung_detect/dataset/label_dataset/'
    data_path='D:/install/Unet/Lung_detect/dataset/cal/'
    data_list=glob.glob(os.path.join(data_path,'*.png'))
    xml_list=glob.glob(os.path.join(data_path,'*.xml'))
    for i in tqdm(range(len(xml_list))):
        xml_name=xml_list[i]
        data_name=data_list[i]
        _coor_=ReadCoorFromXml(xml_name)
        for coor in _coor_:
            cen_x, cen_y = (coor[0] + coor[2]) // 2, (coor[1] + coor[3]) // 2
            # 读取dicom文件的元数据(dicom tags)
            img = cv2.imread(data_name, cv2.IMREAD_GRAYSCALE)
            print('before resize: ', img.shape)
            img_X = ndi.interpolation.zoom(img, [320 / 512, 320 / 512], mode='nearest')
            print('before resize: ', img_X.shape)
            cv2.imwrite(tsave_path + 'chapter3_img_X_{}.png'.format(i), img_X)

            print(cen_x, cen_y)
            rate=320 / 512
            cen_x, cen_y= int(cen_x*rate), int(cen_y*rate)
            img_Y = make_mask((cen_x, cen_y), 3, 320, 320)
            img_Y[img_Y < 0.5] = 0
            img_Y[img_Y > 0.5] = 255
            nodule_mask = img_Y.astype('uint8')
            cv2.imwrite(lsave_path + 'chapter3_img_Y_{}.png'.format(i), nodule_mask)








'D:/install/Unet/Lung_detect/dataset/data/img_'
if __name__ == '__main__':
    Test()
    exit()
    # all_patient_dir='D:/workspace/dataset/libin/DICOM/wuhan_shuju_10/new_patient/case/'
    # listdir(all_patient_dir)#提取病例文件夹
    save_png='D:/install/Unet/Lung_detect/dataset/data/img_'
    with open('path.txt','r') as fp:
        lines=fp.readlines()
        print(lines)
        for line in  lines:
            dicom_dir = line.strip('\n') # 读取dicom文件的元数据(dicom tags)
            extern_name=dicom_dir.split('/')[-1]
            slices = load_patient(dicom_dir)
            print('The number of dicom files : ', len(slices))
            image = get_pixels_hu_by_simpleitk(dicom_dir)  # 提取dicom文件中的像素值
            for i in tqdm(range(image.shape[0])):
                img_path = save_png + extern_name+str(i).rjust(4, '0') + "_i.png"
                org_img = normalize_hu(image[i])*255  # 将像素值归一化到[0,1]区间
                cv2.imwrite(img_path, org_img)  # 保存图像数组为灰度图(.png)



