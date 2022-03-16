import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=0)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.norm_1 = nn.InstanceNorm2d(output_channel)
        self.norm_2 = nn.InstanceNorm2d(output_channel)
        self.upsample = upsample
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='bilinear', scale_factor=2)
        x_s = self.conv_shortcut(x)
        x = self.conv1(self.reflecPad1(x))
        x = self.relu(x)
        x = self.norm_1(x)
        x = self.conv2(self.reflecPad2(x))
        x = self.relu(x)
        x = self.norm_2(x)
        return x_s + x


class decoder_ViT4(nn.Module):
    def __init__(self, isUseShallow=True):
        super(decoder_ViT4, self).__init__()
        self.isUseShallow = isUseShallow
        self.upsample_feat2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.merge_feat12 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(320, 128, (3, 3)),
            nn.InstanceNorm2d(128),
        )
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )  # 64*64
        self.decoder_layer_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 64 if isUseShallow else 256, 256, (3, 3)),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            #nn.Conv2d(256, 256, (3, 3)),
            #nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, feat):
        up_feat2 = self.upsample_feat2(feat[2])

        try:
            merge_feat12 = feat[1] + self.merge_feat12(up_feat2)
        except:
            merge_feat12 = feat[1] + nn.functional.interpolate(
                self.merge_feat12(up_feat2),
                (feat[1].shape[2], feat[1].shape[3]), mode='nearest')

        cs_12 = self.decoder_layer_1(merge_feat12)
        if self.isUseShallow: cs_12 = torch.cat([cs_12, feat[0]], dim=1)
        cs = self.decoder_layer_0(cs_12)
        return cs
