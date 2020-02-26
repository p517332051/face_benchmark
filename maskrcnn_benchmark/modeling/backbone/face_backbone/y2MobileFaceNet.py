# -*- encoding:utf-8 -*-
'''
@Author:Bjj
@Name:y2MobileFaceNet.py
@Data:9/12/19  10:25 AM
@
'''

from torch import nn
import torch


class DResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1):
        super(DResidual, self).__init__()
        self.respart = nn.Sequential(
            nn.Conv2d(in_channels, num_group, kernel_size=(1,1), padding=(0, 0), stride=(1, 1),bias=False),
            nn.BatchNorm2d(num_group),
            nn.PReLU(num_group),
            nn.Conv2d(num_group, num_group, kernel_size=kernel, padding=pad, stride=stride, groups=num_group,bias=False),
            nn.BatchNorm2d(num_group),
            nn.PReLU(num_group),
            nn.Conv2d(num_group, out_channels, kernel_size=(1,1), padding=(0, 0), stride=(1, 1),bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.respart(x)
        return x




class Residual(nn.Module):

    def __init__(self, blocks, in_channels=1,out_channels=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, **kwargs):
        super(Residual, self).__init__()
        self.blocks = blocks
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.DResiduals = self._make_layer(DResidual, in_channels, out_channels,blocks, kernel,stride, pad, num_group)

    def forward(self, x):
        for i in range(self.blocks):
            shortcut = x
            x = self.DResiduals[i](x)
            x = x+shortcut
        return x

    def _make_layer(self, block, in_channels, out_channels, blocks, kernel,stride, pad, num_group):
        layer = []
        layer.append(block(in_channels, out_channels, kernel, stride, pad, num_group= num_group))
        in_channels = out_channels
        for i in range(1,blocks):
            layer.append(block(in_channels, out_channels, kernel, stride=stride, pad=pad, num_group= num_group))
        return nn.Sequential(*layer)


class y2MobileFaceNet(nn.Module):
    def __init__(self,input_size,feature_dim,blocks=[2,8,16,4],**kwargs):
        super(y2MobileFaceNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            Residual(blocks[0], 64, 64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=64)
            )

        self.layer2 = nn.Sequential(
            DResidual(64, 64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128),
            Residual(blocks[1], 64, 64,kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128),
        )
        self.layer3 = nn.Sequential(
            DResidual(64, 128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256),
            Residual(blocks[2], 128, 128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256),
        )

        self.layer4 = nn.Sequential(
            DResidual(128, 128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512),
            Residual(blocks[3], 128, 128, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=256),
            nn.Conv2d(128, 512, kernel_size=(1, 1), padding=(0, 0), stride=1,bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )
        self.linear = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(7,7), stride=(1,1), padding = (0,0), groups=512,bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, feature_dim, kernel_size=(1,1), stride=(1,1), padding = (0,0),bias=False),
            nn.BatchNorm2d(feature_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.size())
        x = self.linear(x)
        x = x.view(x.size(0),-1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)
                # nn.init.xavier_uniform_(m.weight.data,mode='fan_out',gain=2)
                torch.nn.init.kaiming_uniform_(m.weight.data,mode='fan_out',nonlinearity ='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm1d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

def y2(cfg):
    input_size = cfg.INPUT.SIZE_TRAIN
    model = y2MobileFaceNet(input_size=input_size,feature_dim=cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS)
    return model


if __name__ == '__main__':
    input = torch.ones((1,3,112,112))
    model = y2MobileFaceNet().eval()
    print(model(input))