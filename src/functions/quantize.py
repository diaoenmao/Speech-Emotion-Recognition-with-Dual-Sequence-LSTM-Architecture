import torch
from torch.autograd.function import Function

class Quantize(Function):
    def __init__(self):
        super(Quantize, self).__init__()   
        
    @staticmethod
    def forward(ctx, input, num_level, is_training):
        x = input.clone()*(num_level-1)
        if is_training:
            prob = input.new(input.size()).uniform_()
            floor,ceil = x.floor(),x.ceil()
            x[(x-floor)>=prob] = ceil[(x-floor)>=prob]
            x[(x-floor)<prob] = floor[(x-floor)<prob]
        else:
            x = x.round()
        x = x/(num_level-1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None