import torch
from utils import _ntuple

def pixel_unshuffle(input, downscale_factor):
    _tuple = _ntuple(2)
    downscale_factor = _tuple(downscale_factor)
    batch_size, in_channel, in_height, in_width = input.size()
    out_channel = in_channel*(downscale_factor[0]*downscale_factor[1])
    out_height = in_height//downscale_factor[0]
    out_width = in_width//downscale_factor[1]
    input_view = input.reshape(batch_size,in_channel,out_height,downscale_factor[0],out_width,downscale_factor[1])        
    shuffle_out = input_view.permute(0,1,3,5,2,4).reshape(batch_size,out_channel,out_height,out_width)
    return shuffle_out
  
def pixel_shuffle(input, upscale_factor):
    _tuple = _ntuple(2)
    upscale_factor = _tuple(upscale_factor)
    batch_size, in_channel, in_height, in_width = input.size()
    out_channel = in_channel//(upscale_factor[0]*upscale_factor[1])
    out_height = in_height*upscale_factor[0]
    out_width = in_width*upscale_factor[1]
    input_view = input.reshape(batch_size,out_channel,upscale_factor[0],upscale_factor[1],in_height,in_width)        
    shuffle_out = input_view.permute(0,1,4,2,5,3).reshape(batch_size,out_channel,out_height,out_width)      
    return shuffle_out