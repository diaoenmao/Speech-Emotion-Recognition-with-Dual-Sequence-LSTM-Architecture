import codecs
import gzip
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import makedir_exist_ok
from .utils import download_url, make_branch_classes_to_labels

class MNIST(Dataset):
    data_name = 'MNIST'
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0','1','2','3','4','5','6','7','8','9']
    feature_dim = {'img':1}
    output_names = ['img','label']
    
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train        
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.img, self.label = torch.load(os.path.join(self.processed_folder, data_file))
        self.classes_size = 10
        self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        
    def __getitem__(self, index):
        img, label = self.img[index], self.label[index]
        img = Image.fromarray(img.numpy(), mode='L')       
        input = {'img': img, 'label': label}
        if self.transform is not None:
            input = self.transform(input)            
        return input
        
    def __len__(self):
        return len(self.img)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        if self._check_exists():
            return
        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)
        print('Processing...')
        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        print('Done!')

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

class EMNIST(MNIST):
    data_name = 'EMNIST'
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
    digits_classes = ['0','1','2','3','4','5','6','7','8','9']
    upper_letters_classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    lower_letters_classes = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    merged_classes = ['c','i','j','k','l','m','o','p','s','u','v','w','x','y','z']
    feature_dim = {'img':1}
    output_names = ['img','label']
        
    def __init__(self, root, split, branch, **kwargs):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(split, ', '.join(self.splits)))
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super(EMNIST, self).__init__(root, **kwargs)
        self.split = split
        self.branch = branch
        if(self.split == 'digits' or self.split == 'mnist'):
            self.classes = self.digits_classes
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        elif(self.split == 'letters'):
            self.classes = self.upper_letters_classes
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        elif(self.split == 'balanced' or self.split == 'bymerge'):
            unmerged_classes = [c for c in self.lower_letters_classes if c not in self.merged_classes]
            self.classes = self.digits_classes + self.upper_letters_classes + unmerged_classes
            if(branch):
                self.branch_classes = {'digits':self.digits_classes,'letters':self.upper_letters_classes + unmerged_classes}
                self.classes_to_labels, self.classes_to_branch_labels, self.classes_size, self.depth = make_branch_classes_to_labels(self.branch_classes)
            else:
                self.classes_size = len(self.classes)
                self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
        elif(self.split == 'byclass'):
            self.classes = self.digits_classes + self.upper_letters_classes + self.lower_letters_classes
            if(branch):
                self.branch_classes = {'digits':self.digits_classes,'letters':{'upper':self.upper_letters_classes,'lower':self.lower_letters_classes}}
                self.classes_to_labels, self.classes_to_branch_labels, self.classes_size, self.depth = make_branch_classes_to_labels(self.branch_classes)
            else:
                self.classes_size = len(self.classes)
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
            img = Image.fromarray(img.numpy().T, mode='L')
            input = {'img': img, 'label': flat_label, 'branch_label': branch_label}
        else:
            img, label = self.img[index], torch.tensor(self.label[index])
            img = Image.fromarray(img.numpy().T, mode='L')   
            input = {'img': img, 'label': label}
        if self.transform is not None:
            input = self.transform(input)            
        return input
        
    @staticmethod
    def _training_file(split):
        return 'training_{}.pt'.format(split)

    @staticmethod
    def _test_file(split):
        return 'test_{}.pt'.format(split)

    def download(self):
        import shutil
        import zipfile

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.raw_folder, filename)
        download_url(self.url, root=self.raw_folder, filename=filename, md5=None)

        print('Extracting zip archive')
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(self.raw_folder)
        os.unlink(file_path)
        gzip_folder = os.path.join(self.raw_folder, 'gzip')
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith('.gz'):
                self.extract_gzip(gzip_path=os.path.join(gzip_folder, gzip_file))

        for split in self.splits:
            print('Processing ' + split)
            training_set = (
                read_image_file(os.path.join(gzip_folder, 'emnist-{}-train-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(gzip_folder, 'emnist-{}-train-labels-idx1-ubyte'.format(split)))
            )
            test_set = (
                read_image_file(os.path.join(gzip_folder, 'emnist-{}-test-images-idx3-ubyte'.format(split))),
                read_label_file(os.path.join(gzip_folder, 'emnist-{}-test-labels-idx1-ubyte'.format(split)))
            )
            with open(os.path.join(self.processed_folder, self._training_file(split)), 'wb') as f:
                torch.save(training_set, f)
            with open(os.path.join(self.processed_folder, self._test_file(split)), 'wb') as f:
                torch.save(test_set, f)
        shutil.rmtree(gzip_folder)

        print('Done!')

        
class FashionMNIST(MNIST):
    data_name = 'FashionMNIST'
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    feature_dim = {'img':1}
    output_names = ['img','label']

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)