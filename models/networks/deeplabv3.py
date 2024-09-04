import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from ..networks.aspp import build_aspp
from ..networks.decoder import build_decoder
from ..networks.backbone import build_backbone



class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False, inversion=False,contrast=True):
        super(DeepLab, self).__init__()
        self.contrast=contrast
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        layers=list(build_backbone(backbone,output_stride,BatchNorm).children())
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        # assert not np.isnan(x.cpu().detach())
        # assert not np.isnan(low_level_feat.cpu().detach())
        #不像是这
        x = self.aspp(x)
        feature = x
        x1, x2, feature_last = self.decoder(x, low_level_feat)

        x2 = F.interpolate(x2, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)
        #return x1, x2, feature_last,feature
        if self.contrast:
            return feature_last,torch.sigmoid(x1)

        return x1, x2, feature_last
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(),'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]
def DEEPLABV3(inversion=False,contrast=True):
    model=DeepLab(num_classes=2, backbone='mobilenet', output_stride=16,
            sync_bn=False, freeze_bn=False,inversion=inversion,contrast=contrast).cuda()
    return model
if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


