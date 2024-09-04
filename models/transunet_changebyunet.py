from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ComplexConvolutionalBlock import *
from models.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.transunet.transunet import *
from models.GTDLAmodel.vit_seg_modeling import VisionTransformer

from models.GTDLAmodel.vit_seg_configs import get_b16_config
from models.GTDLAmodel.Trans_model import *
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


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
class Down_2(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down_2, self).__init__(
            nn.MaxPool2d(2, stride=2),
            ComplexConvolutionalBlock(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Up_2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_2, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ComplexConvolutionalBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ComplexConvolutionalBlock(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = ComplexConvolutionalBlock(in_channels, base_c)

        factor = 2 if bilinear else 1

        self.down1 = Down_2(base_c, base_c * 2)
        self.down2 = Down_2(base_c * 2, base_c * 4)
        self.down3 = Down_2(base_c * 4, base_c * 8)
        self.down4 = Down_2(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.out_conv = OutConv(base_c, num_classes)

        self.in_conv2 = nn.Conv2d(base_c*2, base_c,kernel_size=1)

        self.down1_2 = Down(base_c, base_c * 2)
        self.down2_2 = Down(base_c * 2, base_c * 4)
        self.down3_2 = Down(base_c * 4, base_c * 8)
        self.down4_2 = Down(base_c * 8, base_c * 16 // factor)
        self.up1_2 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_2 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_2 = Up(base_c * 2, base_c , bilinear)
        self.out_conv_2 = OutConv(base_c, num_classes)


        self.in_conv3 = nn.Conv2d(base_c * 3, base_c,kernel_size=1)
        self.down1_3 = Down(base_c, base_c * 2)
        self.down2_3 = Down(base_c * 2, base_c * 4)
        self.down3_3 = Down(base_c * 4, base_c * 8)
        self.down4_3 = Down(base_c * 8, base_c * 16 // factor)
        self.up1_3 = Up_2(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_3 = Up_2(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_3 = Up_2(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_3 = Up_2(base_c * 2, base_c, bilinear)
        self.out_conv_3 = OutConv(base_c, num_classes)

        self.Up8x8 = up_conv8(ch_in=128,ch_out=1)
        self.Up4x4 = up_conv4(ch_in=64,ch_out=1)

        self.trans = VisionTransformer(get_b16_config())
        self.fconv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.shallow_fusion = shallow_fea_fusion(F_g=32,F_l=32,F_int=32)
        self.l_at1 = local_attention(32)
        self.l_at2 = local_attention(64)
        self.l_at3 = local_attention(128)
        self.l_at4 = local_attention(256)
        self.l_at5 = local_attention(256)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 改编码器
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)  # x9:32,512,512

        logits_1 = self.out_conv(x9)

        x1_2 = self.in_conv2(torch.cat((x1, x9), 1))
        #
        #
        x2_2 = self.down1_2(x1_2)
        x3_2 = self.down2_2(x2_2)
        x4_2 = self.down3_2(x3_2)
        x5_2 = self.down4_2(x4_2)
        x6_2 = self.up1_2(x5_2, x4_2)
        x7_2 = self.up2_2(x6_2, x3_2)
        x8_2 = self.up3_2(x7_2, x2_2)
        x9_2 = self.up4_2(x8_2, x1_2)

        logits_2 = self.out_conv_2(x9_2)

        x1_3 = self.in_conv3(torch.cat((x1, x9, x9_2), 1))
        x2_3 = self.down1_3(x1_3)
        x3_3 = self.down2_3(x2_3)
        x4_3 = self.down3_3(x3_3)
        x5_3 = self.down4_3(x4_3)

        lt5 = self.l_at5(x5_3)
        x5_3 = lt5
        lt4 = self.l_at4(x4_3)
        x4_3 = lt4
        lt3 = self.l_at3(x3_3)
        x3_3 = lt3
        lt2 = self.l_at2(x2_3)
        x2_3 = lt2
        lt1 = self.l_at1(x1_3)
        x1_3 = lt1

        x6_3 = self.up1_3(x5_3, x4_3)
        x7_3 = self.up2_3(x6_3, x3_3)
        x8_3 = self.up3_3(x7_3, x2_3)
        x9_3 = self.up4_3(x8_3, x1_3)

        m6 = self.Up8x8(x6_3)
        m7 = self.Up4x4(x7_3)
        deep_fea = m6 + m7

        shallow_fea = self.shallow_fusion(x8_3, x9_3)
        logits_3 = self.fconv(deep_fea + shallow_fea)


        return logits_1, logits_2, logits_3
