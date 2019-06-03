import torchvision.transforms.functional as F
from PIL import Image

def to_tensor(img):
    return F.to_tensor(img)

def normalize(input,stats):
    for k in input:
        if(k != 'label'):
            tensor = input[k]
            transposed_tensor = tensor.transpose(0,stats[k].feature_dim-1)
            for t, m, s in zip(transposed_tensor, stats[k].mean, stats[k].std):
                t.sub_(m).div_(s)
            input[k] = transposed_tensor.transpose(0,stats[k].feature_dim-1)
    return input
            
    
def resize(img, shape, interpolation):
    W, H = img.size
    img = F.resize(img, shape, interpolation)
    return img

def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
    
def hflip_bbox(bbox,shape):
    H, _ = shape
    H_max = H - bbox[:, 0]
    H_min = H - bbox[:, 2]
    bbox[:, 0] = H_min
    bbox[:, 2] = H_max
    return bbox
    
def vflip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)
    
def vflip_bbox(bbox,shape):
    _, W = shape
    W_max = W - bbox[:, 1]
    W_min = W - bbox[:, 3]
    bbox[:, 1] = W_min
    bbox[:, 3] = W_max
    return bbox
    
def fraction_resize(img, max_shape, interpolation):
    W, H = img.size
    max_H, max_W = max_shape
    scale_H = max_H // H
    scale_W = max_W // W
    if(scale_H >= scale_W):
        img = F.resize(img, (H*scale_W, max_W), interpolation)
    else:
        img = F.resize(img, (max_H, W*scale_H), interpolation)
    return img
    
    
def bbox_resize(bbox, input_shape, output_shape):
    input_W, input_H = input_shape
    output_W, output_H = output_shape
    H_scale = float(output_H) / input_H
    W_scale = float(output_W) / input_W
    bbox[:, 0] = H_scale * bbox[:, 0]
    bbox[:, 2] = H_scale * bbox[:, 2]
    bbox[:, 1] = W_scale * bbox[:, 1]
    bbox[:, 3] = W_scale * bbox[:, 3]
    return bbox