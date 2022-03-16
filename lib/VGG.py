import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(3, 3, (1, 1))]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.ReflectionPad2d((1, 1, 1, 1)),
                           nn.Conv2d(in_channels, x, kernel_size=3),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


from collections import namedtuple

vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])


class loss_network(nn.Module):
    def __init__(self, VGG, requires_grad=False):
        super(loss_network, self).__init__()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), VGG[x])
        for x in range(4, 11):
            self.slice2.add_module(str(x), VGG[x])
        for x in range(11, 18):
            self.slice3.add_module(str(x), VGG[x])
        for x in range(18, 31):
            self.slice4.add_module(str(x), VGG[x])
        for x in range(31, 44):
            self.slice5.add_module(str(x), VGG[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1)
        return out


class VGG_Feature(loss_network):
    def __init__(self, VGG, requires_grad=False):
        super(VGG_Feature, self).__init__(VGG, requires_grad)
        for param in self.parameters():
            param.requires_grad = False

        self.adj_channel_0 = nn.Sequential(nn.Conv2d(256, 64, 1,1,0),
                                           nn.ReLU())
        self.adj_channel_1 = nn.Sequential(nn.Conv2d(512, 128, 1,1,0),
                                         nn.ReLU())
        self.adj_channel_2 = nn.Sequential(nn.Conv2d(512, 320, 1,1,0),
                                           nn.ReLU())
        return

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h
        return [self.adj_channel_0(h_relu3_1), self.adj_channel_1(h_relu4_1), self.adj_channel_2(h_relu5_1)]


class VGG_Decoder(nn.Module):
    def __init__(self):
        super(VGG_Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, latent_code):
        return self.net(latent_code)


if __name__ == "__main__":
    vgg_net = VGG('VGG19')
    vgg_net = torch.nn.Sequential(*list(vgg_net.features.children())[:44])
    net = loss_network(vgg_net)
    input = torch.rand((1,3,256,256))
    output = net(input)
    for item in output:
        print(item.shape)
