import torch
import torch.nn as nn
from torch.autograd import Variable
# from torchvision.models import resnet
import torch.nn.functional as F
# from .scse_original import SCSEBlock
from .resnet import resnet34, SPBlock
from .eca_module import ECABlock

class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        self.spm = SPBlock(channel, channel, norm_layer=nn.BatchNorm2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        spm_out = self.spm(x)
        out = (x + dilate1_out + dilate2_out + dilate3_out + dilate4_out) * spm_out
        #         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x

# class Multiscale_Attention(nn.Module):
#     '''
#     单特征 进行通道加权,作用类似SE模块
#     '''
#
#     def __init__(self, in_channels, out_channels):
#         super(Multiscale_Attention, self).__init__()
#
#         self.local_att = nn.Sequential(
#             nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channels),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#         # self.SCSEBlockMS = SCSEBlock(128)
#
#     def forward(self, x):
#         xl = self.local_att(x)
#         xg = self.global_att(x)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         return x * wei

class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        x = x + x_res

        return x

class MSALNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, num_classes=1):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(MSALNet, self).__init__()

        # base = resnet.resnet18(pretrained=True)
        base = resnet34()

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        self.dblock = Dblock(512)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 2, 2, 0)
        # self.lsm = nn.LogSoftmax(dim=1)
        # self.SCSEBlock4 = SCSEBlock(256)
        # self.SCSEBlock3 = SCSEBlock(128)
        # self.SCSEBlock2 = SCSEBlock(64)
        # self.SCSEBlock1 = SCSEBlock(64)

        # self.MS_CAM4 = Multiscale_Attention(512, 256)
        # self.MS_CAM3 = Multiscale_Attention(256, 128)
        # self.MS_CAM2 = Multiscale_Attention(128, 64)
        # self.MS_CAM1 = Multiscale_Attention(64, 64)

        self.ECABlock4 = ECABlock(512)
        self.ECABlock3 = ECABlock(256)
        self.ECABlock2 = ECABlock(128)
        self.ECABlock1 = ECABlock(64)

        self.br = BR(num_classes)


    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # center block
        e4 = self.dblock(e4)

        # Efficient channel Attention Block with Decoder

        d4 = self.decoder4(e4) + e3
        d4 = self.ECABlock4(d4)
        d3 = self.decoder3(d4) + e2
        d3 = self.ECABlock3(d3)
        d2 = e1 + F.upsample(self.decoder2(d3), (e1.size(2), e1.size(3)), mode='bilinear')
        #d2 = self.decoder2(d3) + e1
        d2 = self.ECABlock2(d2)
        d1 = self.decoder1(d2) + x
        d1 = self.ECABlock1(d1)

        # Multiscale Channel Attention Block with Decoder

        # d4 = self.decoder4(e4) + e3
        # d4 = self.MS_CAM4(d4)
        # d3 = self.decoder3(d4) + e2
        # d3 = self.MS_CAM3(d3)
        # d2 = e1 + F.upsample(self.decoder2(d3), (e1.size(2), e1.size(3)), mode='bilinear')
        # #d2 = self.decoder2(d3) + e1
        # d2 = self.MS_CAM2(d2)
        # d1 = self.decoder1(d2) + x
        # d1 = self.MS_CAM1(d1)

        # Sequeeze Excitation Block with Decoder

        # d4 = self.decoder4(e4) + e3
        # d4 = self.SCSEBlock4(d4)
        # d3 = self.decoder3(d4) + e2
        # d3 = self.SCSEBlock3(d3)
        # d2 = e1 + F.upsample(self.decoder2(d3), (e1.size(2), e1.size(3)), mode='bilinear')
        # #d2 = self.decoder2(d3) + e1
        # d2 = self.SCSEBlock2(d2)
        # d1 = self.decoder1(d2) + x
        # d1 = self.SCSEBlock1(d1)



        #d4 = e3 + self.decoder4(e4)
        # d4 = e3 + self.decoder4(e4)
        # d3 = e2 + self.decoder3(d4)
        # d2 = e1 + self.decoder2(d3)
        # d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        y = self.br(y)

        # y = self.lsm(y)

        return F.sigmoid(y)

