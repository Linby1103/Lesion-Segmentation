# @File  : Get_Segment_Lung.py
# @Author: LiBin
# @Date  : 2020/3/19
# @Desc
# :
import SimpleITK
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
import cv2
from tqdm import tqdm
from dataset import Dicom2JPEG

def get_segmented_lungs(im):
    '''
    对输入的图像进行肺部区域分割，提取有效的肺部区域，用于模型训练
    :param 输入的图像
    :return: 返回分割结果
    '''

    binary = im < -400  # Step 1: 转换为二值化图像
    cleared = clear_border(binary)  # Step 2: 清除图像边界的小块区域
    label_image = label(cleared)  # Step 3: 分割图像

    areas = [r.area for r in regionprops(label_image)]  # Step 4: 保留2个最大的连通区域
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    selem = disk(2)  # Step 5: 图像腐蚀操作,将结节与血管剥离
    binary = binary_erosion(binary, selem)
    selem = disk(10)  # Step 6: 图像闭环操作,保留贴近肺壁的结节
    binary = binary_closing(binary, selem)
    edges = roberts(binary)  # Step 7: 进一步将肺区残余小孔区域填充
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0  # Step 8: 将二值化图像叠加到输入图像上
    im[get_high_vals] = -2000
    print('lung segmentation complete.')
    return im, binary

if __name__ == '__main__':
    dicom_dir = 'D:/workspace/dataset/libin/DICOM/1.3.12.2.1107.5.1.4.86790.30000020012114354701200034455__LungCT__-600__1200/111'
    # 提取dicom文件中的像素值
    save_path='D:/install/Unet/Lung_detect/dataset/Segment_data/'
    image = Dicom2JPEG.get_pixels_hu_by_simpleitk(dicom_dir)      #图片像素提取

    for i in tqdm(range(image.shape[0])):

        label_name_file=save_path+'label'+str(i).rjust(4, '0') + "_i.png"
        train_name_file=save_path+'train'+str(i).rjust(4, '0') + "_i.png"

        im, binary = get_segmented_lungs(image[i])       #肺部区域分割
        org_img = Dicom2JPEG.normalize_hu(image[i])                 #【0,1】归一化
        cv2.imwrite(label_name_file, org_img*255)
        cv2.imwrite(train_name_file, binary*255)
