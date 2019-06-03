import numpy as np
import os
import pickle
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import makedir_exist_ok
from .utils import download_url, check_integrity, make_branch_classes_to_labels

class CIFAR10(Dataset):
    data_name = 'CIFAR10'
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }    
    classes =[
        'airplane', 
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']
    feature_dim = {'img':1}
    output_names = ['img','label']
    
    def __init__(self, root, train=True,
                 transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.img = []
        self.label = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.img.append(entry['data'])
                if 'labels' in entry:
                    self.label.extend(entry['labels'])
                else:
                    self.label.extend(entry['fine_labels'])

        self.img = np.vstack(self.img).reshape(-1, 3, 32, 32)
        self.img = self.img.transpose((0, 2, 3, 1))  # convert to HWC       
        self._load_meta()
        self.classes_size = 10 
        self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        
    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        
    def __getitem__(self, index):
        img, label = self.img[index], torch.tensor(self.label[index])
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        input = {'img': img, 'label': label}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.img)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

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


class CIFAR100(CIFAR10):
    data_name = 'CIFAR100'
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm']
    feature_dim = {'img':1}
    output_names = ['img','label']
    
    def __init__(self, root, branch, **kwargs):
        super(CIFAR100, self).__init__(root, **kwargs)
        self.branch = branch                         
        if(branch):
            self.branch_classes = branch_classes
            self.classes_to_labels, self.classes_to_branch_labels, self.classes_size, self.depth = make_branch_classes_to_labels(self.branch_classes)
        else:
            self.classes_size = 100 
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            
    def __getitem__(self, index):
        if(self.branch):
            img, label = self.img[index], self.label[index]
            flat_label = self.classes_to_labels[self.classes[label]]
            branch_label = self.classes_to_branch_labels[self.classes[label]]
            flat_label = torch.tensor(flat_label)
            branch_label = torch.tensor(branch_label)
            pad = torch.tensor([-100] * (self.depth - branch_label.size(0)),dtype=torch.int64)
            branch_label = torch.cat((branch_label,pad),0)
            img = Image.fromarray(img)
            input = {'img': img, 'label': flat_label, 'branch_label': branch_label}
        else:
            img, label = self.img[index], torch.tensor(self.label[index])
            img = Image.fromarray(img)
            input = {'img': img, 'label': label}            
        if self.transform is not None:
            input = self.transform(input)            
        return input
        
branch_classes = {
                'aquatic mammals':['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                'fish':['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                'flowers':['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                'food containers':['bottle', 'bowl', 'can', 'cup', 'plate'],
                'fruit and vegetables':['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                'household electrical devices':['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                'household furniture':['bed', 'chair', 'couch', 'table', 'wardrobe'],
                'insects':['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                'large carnivores':['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                'large man-made outdoor things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
                'large natural outdoor scenes':['cloud', 'forest', 'mountain', 'plain', 'sea'],
                'large omnivores and herbivores':['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                'medium-sized mammals':['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                'non-insect invertebrates':['crab', 'lobster', 'snail', 'spider', 'worm'],
                'people':['baby', 'boy', 'girl', 'man', 'woman'],
                'reptiles':['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                'small mammals':['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                'trees':['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                'vehicles 1':['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                'vehicles 2':['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
                }
