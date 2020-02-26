import torch
from torch import nn

def BUILD_Dircm():

    return None




class Discriminator(nn.Module):
    def __init__(self,min_scale,in_channels,out_channels = 256,extra_layer = 2):
        super(Discriminator, self).__init__()
        model = []

        model.append(
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            )
        )
        model.append(nn.BatchNorm2d(in_channels))
        model.append(nn.ReLU())
        in_channels = out_channels
        for i in range(extra_layer):
            model.append(
                    nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        )
            )
            model.append(nn.BatchNorm2d(in_channels))
            model.append(nn.ReLU())
        model.append(nn.Conv2d(in_channels=in_channels,out_channels=1, kernel_size=1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*model)

    def forward(self, features):
        validity = self.model(features)
        return validity
        # validity = nn.AdaptiveAvgPool2d(1, )(validity)
        # return validity.mean()