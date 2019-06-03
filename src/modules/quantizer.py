import config
import torch
import torch.nn as nn
from functions import Quantize as QuantizeFunction

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()
        if(config.PARAM['num_levels']==1):
            self.quantizer = nn.Sequential()
        elif(config.PARAM['num_levels']>=2):
            self.quantizer = Quantize(config.PARAM['num_levels'])
        else:
            raise ValueError('Invalid number of quantization level')

    def forward(self, input):
        x = input
        x = self.quantizer(x)
        return x
    
class Quantize(nn.Module):
    def __init__(self,num_levels):
        super().__init__()
        self.num_levels = num_levels
        
    def forward(self, x):
        return QuantizeFunction.apply(x,self.num_levels,self.training)

