import torch
import torch.nn as nn
from functions import pixel_unshuffle, pixel_shuffle

class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        x = pixel_unshuffle(input, self.downscale_factor)
        return x
        
    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)
      
class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        x = pixel_shuffle(input, self.upscale_factor)
        return x
        
    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)