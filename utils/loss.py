# -*- encoding: utf-8 -*-
# Time        :2020/12/19 21:17:50
# Author      :Chen
# FileName    :loss.py
# Version     :1.0
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


# iou loss
def iou_loss(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return 1 - iou_score


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, weight=np.array([1, 1, 1]), is_sigmoid=True):
        # short[1,0,0]、long[0,1,0]
        weight = torch.from_numpy(weight)
        smooth = 1
        dice_score = 0
        dice_loss = 0
        size = pred.size(0)
        if is_sigmoid:
            pred = torch.sigmoid(pred)
        pred_flat_begin = pred.view(size, -1)
        target_flat_begin = target.view(size, -1)
        if weight.sum() != torch.tensor(-1):
            for i in range(pred.shape[0]):
                # short_weight_map = (torch.eq(tiny[i], 0).long()) * weight[i][0] + (torch.eq(tiny[i], 1).long()) * weight[i][2]
                # # long_weight_map = (torch.eq(huge[i], 0).long()) * weight[i][0] + (torch.eq(huge[i], 1).long()) * weight[i][1]
                # # weight_map = (short_weight_map/weight[i][0]) * (long_weight_map/weight[i][0])
                # weight_map = (short_weight_map/weight[i][0])
                weight_map = weight[i]
                # weight_map=short_weight_map
                weight_map_flat = weight_map.view(-1).cuda()
                pred_flat = pred_flat_begin[i]
                target_flat = target_flat_begin[i]
                weight_flat = weight_map_flat
                assert 255 not in target_flat
                intersection = pred_flat * target_flat
                dice_score__ = (2 * (intersection * weight_flat).sum() + smooth) / (
                            (pred_flat * weight_flat).sum() + (target_flat * weight_flat).sum() + smooth)
                dice_loss__ = 1 - dice_score__.sum()
                dice_loss += dice_loss__
        else:
            for i in range(pred.shape[0]):
                pred_flat = pred_flat_begin[i]
                target_flat = target_flat_begin[i]
                assert 255 not in target_flat
                intersection = pred_flat * target_flat
                dice_score__ = (2 * (intersection).sum() + smooth) / ((pred_flat).sum() + (target_flat).sum() + smooth)
                dice_loss__ = 1 - dice_score__.sum()
                dice_loss += dice_loss__
        dice_loss = dice_loss / size
        assert dice_loss >= 0

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


"""BCE + DICE Loss"""


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


""" Entropy Minimization"""


def jaccard_index(true, pred):
    """计算Jaccard指数，适用于批量图像。

    参数:
    - true: 真实分割的张量，形状为[N, 1, H, W]。
    - pred: 预测分割的张量，形状为[N, 1, H, W]。

    返回:
    - Jaccard指数的列表，每个元素对应批量中的一个图像。
    """
    # 确保输入是布尔类型
    true = true.bool()
    pred = pred.bool()

    # 初始化Jaccard指数列表
    jaccard_scores = []

    # 对批量中的每个图像计算Jaccard指数
    for t, p in zip(true, pred):
        intersection = (t & p).float().sum()  # 计算交集
        union = (t | p).float().sum()  # 计算并集

        # 计算Jaccard指数
        jaccard = intersection / union if union != 0 else torch.tensor(0.0)  # 避免除以0
        jaccard_scores.append(jaccard)

    return jaccard_scores
class softCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1):
        super(softCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions (N, C, H, W)
        :param target: target distribution (N, C, H, W)
        :return: loss
        """
        assert inputs.size() == target.size()
        mask = (target != self.ignore_index)

        log_likelihood = F.log_softmax(inputs, dim=1)
        loss = torch.mean(torch.mul(-log_likelihood, target)[mask])

        return loss


"""Maxsquare Loss"""


class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        # self.num_class = num_class

    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        # label = (prob != self.ignore_index)
        loss = -torch.mean(torch.pow(prob, 2) + torch.pow(1 - prob, 2)) / 2
        return loss

import numpy as np
from ripser import ripser
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize
import torch
import torch.nn.functional as F


def calculate_persistence_diagrams(binary_img, resize_scale=0.5):
    """
    计算给定二值图像的持久性图。
    binary_img 应该是一个二值化的 NumPy 数组。
    resize_scale 是用于降采样的因子。
    """
    # 降采样图像
    if resize_scale != 1:
        binary_img = resize(binary_img,
                            (int(binary_img.shape[0] * resize_scale), int(binary_img.shape[1] * resize_scale)),
                            anti_aliasing=False) > 0.5  # 重新阈值化

    # 提取图像中的点集
    points = np.array(np.where(binary_img)).T
    if len(points) == 0:
        return np.array([])  # 如果图像为空，则返回空数组

    # 计算持久性同调
    diagrams = ripser(points)['dgms']
    return diagrams


def tcc_loss(output, target, alpha=0.5, resize_scale=0.5):
    """
    计算 TCLoss。
    output 和 target 是模型的输出和真实标签，尺寸为 [batch_size, 1, height, width]
    alpha 是权重因子，用于平衡交叉熵和拓扑损失
    resize_scale 是用于图像降采样的比例，减少计算复杂度
    """
    batch_size = output.shape[0]
    total_loss = 0.0
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    # 遍历批量中的每一个样本
    for i in range(batch_size):
        # 二值化并降采样
        binary_output = resize((output[i, 0] > 0.5).astype(float),
                               (int(output.shape[2] * resize_scale), int(output.shape[3] * resize_scale)),
                               anti_aliasing=False) > 0.5
        binary_target = resize((target[i, 0] > 0.5).astype(float),
                               (int(target.shape[2] * resize_scale), int(target.shape[3] * resize_scale)),
                               anti_aliasing=False) > 0.5

        # 计算每个样本的持久性图
        pd_output = calculate_persistence_diagrams(binary_output)
        pd_target = calculate_persistence_diagrams(binary_target)

        # 计算拓扑损失，这里使用 1-dim 的 Wasserstein 距离作为示例
        if len(pd_output) == 0 or len(pd_target) == 0 or len(pd_output[1]) == 0 or len(pd_target[1]) == 0:
            topo_loss = 0
        else:
            topo_loss = directed_hausdorff(pd_output[1], pd_target[1])[0]

        # 计算交叉熵损失
        ce_loss = F.binary_cross_entropy_with_logits(torch.from_numpy(output[i]), torch.from_numpy(target[i]))

        # 结合两种损失
        total_loss += alpha * ce_loss + (1 - alpha) * topo_loss
        # total_loss +=  topo_loss

    return total_loss / batch_size