import cv2
import numpy
import numpy as np
import torch
from PIL import Image
import numpy as np
from PIL import Image
import os
from torch import nn
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label

from skimage import io, morphology
import matplotlib.pyplot as plt


class SK_loss(nn.Module):
    def __init__(self):
        super(SK_loss, self).__init__()

    def forward(self, predict):
        assert torch.unique(predict).shape == 0
        sk_pre = skelton(predict)
        loss = 0
        return loss

        # tensor


def skelton(img):
    img = np.array(img.cpu().detach().numpy()).astype('uint8')
    img = img.squeeze(1)
    x_tr = 0
    batch = img.shape[0]
    height = img.shape[1]
    width = img.shape[2]
    # 使用PIL打开GIF图像
    # im = Image.open(imgpath)
    # 转为RGB格式
    # im = im.convert('RGB')
    # 转换为OpenCV格式的numpy数组
    # result = np.zeros((batch,height, width), dtype=np.uint8)

    skeleton = morphology.skeletonize(img > 0)
    return skeleton


cross_kernel = np.array([[1, 1, 1],
                         [1, 10, 1],
                         [1, 1, 1]], dtype=np.uint8)


def extract_crosspoints(skel):
    # 定义一个3x3的十字形结构元素
    a = np.zeros_like(skel)
    # 应用卷积
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, cross_kernel)
    # 交叉点是卷积结果大于等于30的像素点（中心为1，且至少有三个邻居）
    crosspoints = filtered >= 13
    a[crosspoints] = 1
    return a


def extract_endpoints(skel):
    a = np.zeros_like(skel)

    # 定义一个3x3的十字形结构元素
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, cross_kernel)
    # 端点是卷积结果为11的像素点（中心为1，且只有一个邻居）
    endpoints = filtered == 11
    # 应用卷积
    a[endpoints] = 1

    return a


def remove_small_cc(skeleton, min_size=5):
    skeleton = np.array(skeleton,dtype=bool)
    labeled_skeleton = label(skeleton)

    cleaned_skeleton = remove_small_objects(labeled_skeleton, min_size=min_size)
    return (cleaned_skeleton > 0).astype(int)  # 返回清理后的骨架图


def keypoint(ske):
    # todo:要不要过滤一下
    ske = remove_small_cc(ske)
    endpoints = extract_endpoints(ske)
    crosspoints = extract_crosspoints(ske)

    return endpoints + crosspoints
    # cv2.imwrite('{:0>2d}.tif'.format(i), skell)
    # 显示结果


# cv2.imshow("skeleton", skell)
# cv2.imwrite(filename, img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == '__main__':
    path = r'D:\2021\wxr\datasets\DRIVE\training\1st_manual'
    datalist = os.listdir(path)
    list = [os.path.join(path, i) for i in datalist]
    for i in range(len(list)):
        a = skelton(io.imread(list[i]))
        Image.fromarray(np.array(a)).show()
# 骨架化后修复断点

# 闭运算补连接断点
# kernel = np.ones((5,5),np.uint8)
# close = cv2.morphologyEx(skell, cv2.MORPH_CLOSE, kernel)
#
# # 距离变换连接
# dist = cv2.distanceTransform(skell, cv2.DIST_L2, 3)
#
# markers = np.zeros_like(skell)
# markers[dist<3] = 255
# segmentation = cv2.watershed(dist, markers)
# # 将单通道dist复制三份合成三通道
# dist_3ch = np.dstack([dist]*3)
#
# # 然后作为src传入watershed
# segmentation = cv2.watershed(dist_3ch, markers)
# seg_img = np.where(segmentation == -1, 255, 0).astype(np.uint8)
#
# # 显示修复结果
# cv2.imshow('close', close)
# cv2.imshow('watershed', seg_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()