import os
import sys
import torch
from torch.utils.data import Dataset
from utils import list_dir
from .utils import default_loader, make_img_dataset, make_classes_counts

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
       
class DatasetFolder(Dataset):
    def __init__(self, root, loader, extensions, transform=None):
        self.root = root
        self.loader = loader
        self.extensions = extensions 
        dirs = list_dir(self.root)
        if(dirs != []):
            self.classes, self.classes_to_labels = self._find_classes(self.root)
            self.classes_size = len(self.classes_to_labels.keys())
            self.output_names = ['img','label']
            self.classes_counts = make_classes_counts(self.data['label'],self.classes_size)
        else:
            self.classes_to_labels = None
            self.classes_size = 0
            self.output_names = ['img']
        self.data = make_img_dataset(self.root, self.extensions, self.classes_to_labels)       
        self.transform = transform

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        return classes, classes_to_labels

    def __getitem__(self, index):
        input = {}
        for k in self.output_names:
            if(k == 'img'):
                path = self.data['img'][index]
                img = self.loader(path)
                input['img'] = img
            elif(k == 'label'):
                input['label'] = torch.tensor(self.data['label'][index])
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

class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform)