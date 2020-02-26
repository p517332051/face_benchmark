from maskrcnn_benchmark.modeling.backbone.face_backbone.OctResNet.Octconv import *


__all__ = ['OctResNet', 'oct_resnet26', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200']




class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False,expansion=4):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.expansion = expansion
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv_BN_ACT(inplanes, width, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv3 = Conv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.prelu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.prelu(x_h)
        x_l = self.prelu(x_l) if x_l is not None else None

        return x_h, x_l


        return x_h

class OctResNet(nn.Module):
    def __init__(self,input_size, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, alpha_in=0,expansion=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer,expansion=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer,expansion=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True,expansion=1)
        self.output_layer = nn.Sequential(
                                       # nn.BatchNorm2d(512),
                                       nn.Dropout(),
                                       # nn.Flatten(),
                                       nn.Conv2d(512,512,(int(input_size[0]/16),int(input_size[1]/16)),padding=0, dilation=1,bias=False),
                                       # nn.Linear(512 * int(input_size[0]/16) * int(input_size[1]/16), 512),
                                       nn.BatchNorm2d(512),
                                       nn.Flatten(),
                                       )
        # get_block(in_channel=64, depth=64, num_units=3),
        # get_block(in_channel=64, depth=128, num_units=13),
        # get_block(in_channel=128, depth=256, num_units=30),
        # get_block(in_channel=256, depth=512, num_units=3)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False,expansion = None):
        if expansion==None:
            expansion = block.expansion

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * expansion, kernel_size=1, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output,expansion = expansion))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output,expansion = expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h,x_l))
        x_h, x_l = self.layer3((x_h,x_l))
        x_h, x_l = self.layer4((x_h,x_l))

        x = self.output_layer(x_h)
        return x


def oct_resnet26(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-26 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def oct_resnet50(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def oct_resnet101(cfg, **kwargs):
    """Constructs a Octave ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    input_size = cfg.INPUT.SIZE_TRAIN
    model = OctResNet(input_size,Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def oct_resnet152(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def oct_resnet200(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-200 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OctResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
