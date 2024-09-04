import torch
import torch.nn as nn
import torch.nn.functional as F
BN_EPS = 1e-4  #1e-4  #1e-5
#1.把doubleconv换成这个if融合的
#2.把doubleconv的第二阶段换成这个if融合的
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm
        if is_relu is False: self.relu=None
        # self.swish=Swish() 不行

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        # x = self.swish(x)
        return x
class M_Encoder(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False,
                 BatchNorm=False, num_groups=32,
                 res=False):
        super(M_Encoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode1 = ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                  dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                  num_groups=num_groups)
        self.encode2 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                  dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                  num_groups=num_groups)
        self.encode3 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                  dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                  num_groups=num_groups)

        self.pooling = pooling

        contact_channels = output_channels * 3
        self.se = SE_Block(contact_channels)

        self.conv2 = M_Conv(contact_channels, output_channels, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

    def forward(self, x):

        out1 = self.encode1(x)
        out2 = self.encode2(out1)
        out3 = self.encode3(out2)

        out = torch.cat([out1, out2], dim=1)
        conv = torch.cat([out, out3], dim=1)


        _, ch, _, _ = conv.size()
        conv = self.se(conv)

        conv = self.conv2(conv)

        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            nn.ReLU(inplace=True)(pool)

            return conv, pool
        else:
            return conv, conv

class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        b, c, _, _ = X_input.size()  # shape = [32, 64, 2000, 80]

        y = self.avg_pool(X_input)  # shape = [32, 64, 1, 1]
        y = y.view(b, c)  # shape = [32,64]

        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)  # shape = [32, 64] * [64, 4] = [32, 4]

        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y)  # shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)  # shape = [32, 64, 1, 1]， 这个就表示上面公式的s, 即每个通道的权重

        return X_input * y.expand_as(X_input)

class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv
class coord_attention(nn.Module):
    def __init__(self,channel,reduction=16):
        super(coord_attention, self).__init__()
        self._1x1conv = nn.Conv2d(channel,channel//16,kernel_size=1,stride=1,bias=False)
        self.bn       = nn.BatchNorm2d(channel//reduction)
        self.relu     = nn.ReLU()
        self.F_h      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.F_w      = nn.Conv2d(channel//16,channel,kernel_size=1,stride=1,bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
    def forward(self,x):
        #b,c,h,w
        _,_,h,w = x.size()

        #b,c,h,w->b,c,h,1->b,c,1,h
        x_h = torch.mean(x,dim=3,keepdim=True).permute(0,1,3,2)

        #b,c,h,w->b,c,1,w
        x_w = torch.mean(x,dim=2,keepdim=True)
        x_cat_bn_relu = self.relu(self.bn(self._1x1conv(torch.cat((x_h,x_w),3))))

        x_cat_split_h,x_cat_split_w = x_cat_bn_relu.split([h,w],3)

        s_h =  self.sigmoid_h(self.F_h(x_cat_split_h.permute(0,1,3,2)))
        s_w =  self.sigmoid_w(self.F_w(x_cat_split_w))

        out = x*s_h.expand_as(x)*s_w.expand_as(x)
        return out
class mulite_scale_conv(nn.Sequential):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False,
                 BatchNorm=False, num_groups=32,
                 res=False):
        super(mulite_scale_conv, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode1 = ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                    dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                    num_groups=num_groups)
        self.encode2 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                    dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                    num_groups=num_groups)
        self.encode3 = ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                                    dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                                    num_groups=num_groups)
        contact_channels = output_channels * 3
        self.se = SE_Block(contact_channels)
        self.coord = coord_attention(contact_channels)
        self.conv2 = M_Conv(contact_channels, output_channels, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
    def forward(self, x):
        out1 = self.encode1(x)
        out2 = self.encode2(out1)
        out3 = self.encode3(out2)

        out = torch.cat([out1, out2], dim=1)
        conv = torch.cat([out, out3], dim=1)


        _, ch, _, _ = conv.size()
        conv = self.se(conv)
        # 这个se换成coord试试 ：这个不好使
        # conv = self.coord(conv)



        conv = self.conv2(conv)

        return conv