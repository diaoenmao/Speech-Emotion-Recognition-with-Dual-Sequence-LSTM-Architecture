import numpy as np
import os
import shutil
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import makedir_exist_ok
from .folder import ImageFolder
from .utils import download_url, check_integrity


class CUB2011(ImageFolder):
    data_name = 'CUB2011'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    feature_dim = {'img':1}
    
    def __init__(self, root, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        if(download):
            self.download()
        if(not self._check_exists()):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        super(CUB2011, self).__init__(root, transform)

    def _check_exists(self):
        return os.path.exists(self.root)
            
    def _check_integrity(self):
        md5 = self.tgz_md5
        fpath = os.path.join(os.path.dirname(self.root), self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        import tarfile
        if(not self._check_integrity()):
            download_url(self.url, os.path.dirname(self.root), self.filename, self.tgz_md5)
        if(self._check_exists()):
            print('Files already downloaded and verified')
            return
        else:
            if(not os.path.exists(os.path.join(os.path.dirname(self.root), 'CUB_200_2011'))):
                with tarfile.open(os.path.join(os.path.dirname(self.root), self.filename), "r:gz") as tar:
                    tar.extractall(path=os.path.join(os.path.dirname(self.root), 'CUB_200_2011'))
            image = open(os.path.join(os.path.dirname(self.root), 'CUB_200_2011/CUB_200_2011/images.txt'), 'r')
            split = open(os.path.join(os.path.dirname(self.root), 'CUB_200_2011/CUB_200_2011/train_test_split.txt'), 'r')
            if_train = []
            for line in split.readlines():
                if_train.append(int(line.split()[1]))
            makedir_exist_ok(os.path.join(os.path.dirname(self.root),'train'))
            makedir_exist_ok(os.path.join(os.path.dirname(self.root),'validation'))
            for i in range(len(if_train)):
                line = image.readline()
                image_path = line.split()[1]
                if(if_train[i]==1):
                    makedir_exist_ok(os.path.join(os.path.dirname(self.root),'train',image_path.split('/')[0]))
                    shutil.move(os.path.join(os.path.dirname(self.root),'CUB_200_2011/CUB_200_2011/images',image_path), os.path.join(os.path.dirname(self.root),'train',image_path))
                else:
                    makedir_exist_ok(os.path.join(os.path.dirname(self.root),'validation',image_path.split('/')[0]))
                    shutil.move(os.path.join(os.path.dirname(self.root),'CUB_200_2011/CUB_200_2011/images',image_path), os.path.join(os.path.dirname(self.root),'validation',image_path))
            image.close()
            split.close()
            shutil.rmtree(os.path.join(os.path.dirname(self.root),'CUB_200_2011/CUB_200_2011/images'),ignore_errors=True)
        return
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
