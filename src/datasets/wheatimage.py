import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from .utils import default_loader, make_img_dataset, merge_classes, make_classes_counts

IMG_EXTENSIONS = ['.bmp']

class WheatImage(Dataset):
    data_name = 'WheatImage'
    label_modes = ['binary', 'six']
    output_names = ['img','label']
    feature_dim = {'img':1}
     
    def __init__(self, root, label_mode, transform=None):
        if label_mode not in self.label_modes:
            raise ValueError('label mode not supported')
        self.root = root
        self.label_mode = label_mode
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        if(self.label_mode == 'binary'):
            self.classes = ['cracked','germinant','moldy','mothy','normal','sick']
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            self.data = make_img_dataset(self.root, self.extensions, self.classes_to_labels) 
            self.data['label'] = merge_classes(self.data['label'], {'0':0,'1':0,'2':0,'3':0,'4':1,'5':0})
            self.classes = ['abnormal','normal']
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        elif(self.label_mode == 'six'):
            self.classes = ['cracked','germinant','moldy','mothy','normal','sick']
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            self.data = make_img_dataset(self.root, self.extensions, self.classes_to_labels)
        self.data['label'] = torch.tensor(self.data['label'])
        self.classes_counts = make_classes_counts(self.data['label'],self.classes_size)        
        self.transform = transform
        
    def __getitem__(self, index):
        path, label = self.data['img'][index], self.data['label'][index]
        img = self.loader(path)
        input = {'img': img, 'label': label}
        if self.transform is not None:
            input = self.transform(input)            
        return input

    def __len__(self):
        return len(self.data[self.output_names[0]])
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str