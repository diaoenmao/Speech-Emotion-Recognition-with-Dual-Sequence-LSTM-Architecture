import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

           
class VOCDetection(Dataset):
    classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor']
        
    def __init__(self, root, split, use_difficult=False, transform=None):
        self.root = root
        self.split = split
        self.use_difficult = use_difficult
        self.transform = transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations')  
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '{}.txt'.format(self.split))

        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath + '/{}.jpg'.format(img_id)).convert('RGB')     
        target = ET.parse(self._annopath + '/{}.xml'.format(img_id))              
        bbox = []
        label = []
        difficult = []
        for obj in target.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.classes.index(name))
        bbox = torch.from_numpy(np.stack(bbox).astype(np.float32))
        label = torch.tensor(label)
        difficult = torch.tensor(difficult)        
        input = {'img': img, 'bbox': bbox, 'label': label, 'difficult' :difficult}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.ids)

class VOCSegmentation(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'SegmentationClass')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Segmentation', '{}.txt'.format(self.split))

        with open(self._imgsetpath) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath + '/{}.jpg'.format(img_id)).convert('RGB')     
        segmentation = torch.from_numpy(np.array(Image.open(self._annopath + '/{}.png'.format(img_id))).astype(np.int64))
        segmentation[segmentation==255] = 0
        input = {'img': img, 'segmentation': segmentation}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.ids)
        
        