import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
from .attentons import *
from models.desconv.dcn_module import *
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                        padding=padding, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        return self.relu(x)


class DiagonalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, diagonal_type='main'):
        super(DiagonalConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.diagonal_type = diagonal_type

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.set_diagonal()

    def set_diagonal(self):
        with torch.no_grad():
            self.weight.fill_(0)
            if self.diagonal_type == 'main':
                for i in range(min(self.in_channels, self.out_channels)):
                    idxs = torch.arange(self.kernel_size)
                    self.weight[i, i, idxs, idxs] = torch.randn(self.kernel_size)
            elif self.diagonal_type == 'anti':
                for i in range(min(self.in_channels, self.out_channels)):
                    idxs = torch.arange(self.kernel_size)
                    # 使用临时Tensor赋值
                    temp_tensor = torch.randn(self.kernel_size)
                    self.weight[i, i, idxs, torch.arange(self.kernel_size - 1, -1, -1)] = temp_tensor

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.kernel_size // 2)
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DoubleConv_nobnrelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_nobnrelu, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )


class AdaptiveFusionLayer(nn.Module):
    def __init__(self, num_features, out_channels):
        super(AdaptiveFusionLayer, self).__init__()
        self.weights_fc = nn.ParameterList([
            nn.Parameter(torch.randn(1, out_channels, 1, 1)) for _ in range(num_features)
        ])
        self.softmax = nn.Softmax(dim=0)
        self.num_features = num_features

    def forward(self, features):
        # Assuming features is a list of feature maps
        weighted_features = []
        weights = [F.relu(weight) for weight in self.weights_fc]  # Ensure non-negative weights
        weights = self.softmax(torch.stack(weights))  # Normalize weights #torch.stack(weights):[5,1,32,1,1]

        for i in range(self.num_features):
            weighted_features.append(features[i] * weights[i])

        fused_features = sum(weighted_features)
        return fused_features

class ComplexConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexConvolutionalBlock, self).__init__()
        self.conv1x5 = nn.Conv2d(in_channels, out_channels, (1, 5), padding=(0, 2))
        self.conv5x1 = nn.Conv2d(in_channels, out_channels, (5, 1), padding=(2, 0))
        self.conv_diag = DiagonalConv2d(in_channels, out_channels, kernel_size=5, diagonal_type='main')
        self.conv_anti_diag = DiagonalConv2d(in_channels, out_channels, kernel_size=5, diagonal_type='anti')
        self.deformable_conv = DeformableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn = nn.BatchNorm2d(out_channels )
        self.relu = nn.ReLU(inplace=True)

        self.adaptive_fusion = AdaptiveFusionLayer(5, out_channels ) #实现自适应特征融合
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        conv1x5_out = self.conv1x5(x)
        conv5x1_out = self.conv5x1(x)
        conv_diag_out = self.conv_diag(x)
        conv_anti_diag_out = self.conv_anti_diag(x)
        deformable_feat = self.deformable_conv(x)

        # Collect features for adaptive fusion
        features = [conv1x5_out, conv5x1_out, conv_diag_out, conv_anti_diag_out, deformable_feat]

        # Adaptive fusion of features
        fused_features = self.adaptive_fusion(features)

        # Apply batch normalization and ReLU
        fused_features = self.bn(fused_features)
        fused_features = self.relu(fused_features)

        # Final convolution layer to refine the output features
        output = self.final_conv(fused_features)

        return output


