import collections
import numbers
import sys
import torch
import torchvision
import datasets.functional as F
from PIL import Image

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable
    
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
        
class ToTensor(object):
        
    def __call__(self, input):
        input['img'] = F.to_tensor(input['img'])
        return input

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    def __init__(self, stats):
        self.stats = stats
        
    def __call__(self, input):
        input = F.normalize(input, self.stats)
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.stats.mean, self.stats.std)
        
class Resize(object):
    def __init__(self, shape, interpolation=Image.BILINEAR):
        assert isinstance(shape, int) or (isinstance(shape, Iterable) and len(shape) == 2)
        self.shape = shape
        self.interpolation = interpolation
        
    def __call__(self, input):
        img = input['img']        
        org_shape = img.size
        input['img'] = F.resize(img, self.shape, self.interpolation)
        if('bbox' in input): 
            input['bbox'] = F.bbox_resize(input['bbox'],org_shape,input['img'].size)
        if('segmentation' in input):
            input['segmentation'].unsqueeze_(0).unsqueeze_(0)
            input['segmentation'] = torch.nn.functional.interpolate(input['segmentation'].float(),size=self.shape,mode='nearest').long()
            input['segmentation'].squeeze_()
        return input

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(shape={0}, interpolation={1})'.format(self.shape, interpolate_str)
        
class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, input):
        if torch.rand(1) < self.p:
            img = input['img']
            shape = (img.size[1],img.size[0])
            input['img'] = F.hflip(img)
            if 'bbox' in input:
                bbox = input['bbox']
                input['bbox'] = F.hflip_bbox(bbox,shape)           
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
        
class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.transform = torchvision.transforms.RandomCrop(size,self.padding,self.pad_if_needed)
        
    def __call__(self, input):
        input['img'] = self.transform(input['img'])
        return input
        
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
        
        
class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, input):
        if torch.rand(1) < self.p:
            img = input['img']           
            shape = (img.size[1],img.size[0])
            input['img'] = F.vflip(img)
            if 'bbox' in input:
                bbox = input['bbox']
                input['bbox'] = F.vflip_bbox(bbox,shape)
        return input

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
        
        
class FractionResize(object):
        
    def __init__(self, max_shape, interpolation=Image.BILINEAR):
        assert isinstance(max_shape, int) or (isinstance(max_shape, Iterable) and len(max_shape) == 2)
        self.max_shape = max_shape
        self.interpolation = interpolation
        
    def __call__(self, input):
        img = input['img']        
        org_shape = img.size
        input['img'] = F.fraction_resize(img,self.max_shape,self.interpolation)
        if('bbox' in input):
            input['bbox'] = F.bbox_resize(input['bbox'],org_shape,input['img'].size)
        if('segmentation' in input):
            input['segmentation'].unsqueeze_(0).unsqueeze_(0)
            input_shape = (input['img'].size[1],input['img'].size[0])
            input['segmentation'] = torch.nn.functional.interpolate(input['segmentation'].float(),input_shape,mode='nearest').long()
            input['segmentation'].squeeze_()
        return input

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(max_shape={0}, interpolation={1})'.format(self.max_shape, interpolate_str)
        
