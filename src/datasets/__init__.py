from .cifar import CIFAR10, CIFAR100
from .cub import CUB2011
from .folder import ImageFolder, DatasetFolder
from .mnist import MNIST, EMNIST, FashionMNIST
from .mosi import MOSI
from .svhn import SVHN
from .transforms import *
from .voc import VOCDetection, VOCSegmentation
from .wheatimage import WheatImage

__all__ = ('MNIST','EMNIST', 'FashionMNIST',
           'CIFAR10', 'CIFAR100', 'SVHN',
           'ImageFolder', 'DatasetFolder',
           'VOCDetection', 'VOCSegmentation',
           'CUB2011',
           'WheatImage','MOSI')