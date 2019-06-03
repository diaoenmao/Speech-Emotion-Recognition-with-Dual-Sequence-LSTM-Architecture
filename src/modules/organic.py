import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _ntuple

class _oConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(_oConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)  

class oConv1d(_oConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        ntuple = _ntuple(1)
        kernel_size = ntuple(kernel_size)
        stride = ntuple(stride)
        padding = ntuple(padding)
        dilation = ntuple(dilation)
        super(oConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                       
    def forward(self, input, permutation):
        weight = torch.einsum('ao,oil->ail',permutation,self.weight)
        return F.conv1d(input=input, weight=self.weight[coordinates[0],coordinates[1],], bias=self.bias[coordinates[0].view(-1)] if(self.bias is not None) else self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class oConv2d(_oConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        ntuple = _ntuple(2)
        kernel_size = ntuple(kernel_size)
        stride = ntuple(stride)
        padding = ntuple(padding)
        dilation = ntuple(dilation)
        super(oConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                       
    def forward(self, input, coordinates):
        return F.conv2d(input=input, weight=self.weight[coordinates[0],coordinates[1],], bias=self.bias[coordinates[0].view(-1)] if(self.bias is not None) else self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class oConv3d(_oConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        ntuple = _ntuple(3)
        kernel_size = ntuple(kernel_size)
        stride = ntuple(stride)
        padding = ntuple(padding)
        dilation = ntuple(dilation)
        super(oConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
                       
    def forward(self, input, permutation):
        weight = torch.einsum('ao,oihwd->aihwd',permutation,self.weight)
        return F.conv3d(input=input, weight=self.weight[coordinates[0],coordinates[1],], bias=self.bias[coordinates[0].view(-1)] if(self.bias is not None) else self.bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)