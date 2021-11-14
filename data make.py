# work 二哥
# 训练集制作（压缩至0.75,0.5,0.25，旋转90,180度角）

import pywt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def cut_36(image, Name):
    """
    对图像进行36*36的切割（间隔14）
    :param image: 输入图像
    :param Name: 图像名称
    :return: 将结果依次保存在train image36中
    """
    [height, width] = image.shape
    t_h = np.floor((height - 33) / 14).astype(int)
    t_w = np.floor((width - 33) / 14).astype(int)
    # print(t_h, t_w)
    num = 0
    for h in range(t_h + 1):
        for w in range(t_w + 1):
            newImage = image[h * 14: h * 14 + 33, w * 14: w * 14 + 33]
            # print(newImage.shape,newImage)
            cv.imwrite('train_image\\'+'train' + Name + '_' + str(num) + '.bmp', newImage)
            num += 1


path_file = '91pictures'
file_name = os.listdir(path_file)
print(file_name)
for i in file_name:
    print(i)
    jpgName = i[:i.find('.bmp')]  # 去掉后缀的文件名
    image = cv.imread(path_file +'\\'+ i, 0)

    # image = cv.imread(path_file + '223060.jpg', 0)
    # jpgName = '223060'
    # print(image)

    [m, n] = image.shape
    cut_36(image, jpgName)
    tem_75 = cv.resize(image, (int(m * 1.12), int(n * 1.12)))
    cut_36(tem_75, jpgName + '_1.25')
    tem_75 = cv.resize(image, (int(m * 6 / 7), int(n * 6 / 7)))
    cut_36(tem_75, jpgName + '_0.75')

    # if np.random.rand()< 0.1:
    #     tem_50 = cv.resize(image,(int(m*2/3),int(n*2/3)))
    #     cut_64(tem_50, jpgName+'_0.50')
    # if np.random.rand()< 0.1:
    #     tem_25 = cv.resize(image,(int(m/2),int(n/2)))
    #     cut_64(tem_25, jpgName+'_0.25')

    tem_x = cv.flip(image, 0)
    cut_36(tem_x, jpgName + '_x')
    tem_y = cv.flip(image, 1)
    cut_36(tem_y, jpgName + '_y')
    tem_xy = cv.flip(image, -1)
    cut_36(tem_xy, jpgName + '_xy')
