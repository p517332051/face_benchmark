import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

class VConv2d(nn.modules.conv._ConvNd):
  """
  Versatile Filters
  Paper: https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, delta=0, g=1,padding_mode='zeros'):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super(VConv2d, self).__init__(
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias,padding_mode =padding_mode)
    self.s_num = int(np.ceil(self.kernel_size[0]/2))  # s in paper
    self.delta = delta  # c-\hat{c} in paper
    self.g = g  # g in paper
    self.weight = nn.Parameter(torch.Tensor(
                int(out_channels/self.s_num/(1+self.delta/self.g)), in_channels // groups, *kernel_size))
    self.reset_parameters()

  def forward(self, x):
    x_list = []
    s_num = self.s_num
    ch_ratio = (1+self.delta/self.g)
    ch_len = self.in_channels - self.delta
    for s in range(s_num):
        for start in range(0, self.delta+1, self.g):
            weight1 = self.weight[:, :ch_len, s:self.kernel_size[0]-s, s:self.kernel_size[0]-s]
            if self.padding[0]-s < 0:
                h = x.size(2)
                x1 = x[:,start:start+ch_len,s:h-s,s:h-s]
                padding1 = _pair(0)
            else:
                x1 = x[:,start:start+ch_len,:,:]
                padding1 = _pair(self.padding[0]-s)

            if self.bias==None:
                bias = self.bias
            else:
                bias = self.bias[int(self.out_channels * (s * ch_ratio + start) / s_num / ch_ratio):int(
                    self.out_channels * (s * ch_ratio + start + 1) / s_num / ch_ratio)]

            x_list.append(F.conv2d(x1, weight1,bias, self.stride,
                      padding1, self.dilation, self.groups))
    x = torch.cat(x_list, 1)
    return x
