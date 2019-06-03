import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import *
from .utils import make_classes_counts

class MOSI(Dataset):
    data_name = 'MOSI'
    label_modes = ['binary','five','seven','regression']
    supported_feature_names = {'covarep':'COVAREP','opensmile':'OpenSmile-emobase2010','facet':'FACET 4.1','glove':'glove_vectors','bert':'BERT embeddings','label':'Opinion Segment Labels'}
    output_names = ['covarep','facet','glove','label']
    feature_dim = {'covarep':2,'opensmile':2,'facet':2,'glove':2,'bert':2}
    
    def __init__(self, root, split, label_mode, transform=None, download=False):
        if label_mode not in self.label_modes:
            raise ValueError('label mode not found')
        self.root = root
        self.split = split
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if(self.split=='train'):
            self.data = load(os.path.join(self.root, 'processed', 'train.pt'))
        elif(self.split=='val'):
            self.data = load(os.path.join(self.root, 'processed', 'validation.pt'))
        elif(self.split=='trainval'):
            train_data = load(os.path.join(self.root, 'processed', 'train.pt'))
            validation_data = load(os.path.join(self.root, 'processed', 'validation.pt'))
            for k in train_data:
                train_data[k].extend(validation_data[k])
            self.data = train_data            
        elif(self.split=='test'):
            self.data = load(os.path.join(self.root, 'processed', 'test.pt'))
        else:
            raise ValueError('Data split not supported')
        self.data['label'] = torch.tensor(self.data['label'])
        if(label_mode == 'binary'):
            self.classes = ['negative','positive']
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            self.data['label'][self.data['label'] >= 0] = 1
            self.data['label'][self.data['label'] < 0] = 0
            self.data['label'] = self.data['label'].long()
            self.classes_counts = make_classes_counts(self.data['label'],self.classes_size)
        elif(label_mode == 'five'):
            self.classes = ['negative','somewhat negative','neutral','somewhat positive','positive']
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            self.data['label'] = torch.round(self.data['label']/3*2 + 2).long()
            self.classes_counts = make_classes_counts(self.data['label'],self.classes_size)
        elif(label_mode == 'seven'):
            self.classes = ['very negative','negative','somewhat negative','neutral','somewhat positive','positive','very positive']
            self.classes_size = len(self.classes)
            self.classes_to_labels = {self.classes[i]:i for i in range(len(self.classes))}
            self.data['label'] = torch.round(self.data['label'] + 3).long()
            self.classes_counts = make_classes_counts(self.data['label'],self.classes_size)
        elif(label_mode == 'regression'):
            pass
        else:
            raise ValueError('label mode not supported')
        self.transform = transform
        
    def __len__(self):
        return len(self.data[self.output_names[0]])

    def __getitem__(self, idx):
        input = {}
        for k in self.output_names:
            input[k] = torch.tensor(self.data[k][idx]) if(not isinstance(self.data[k][idx], torch.Tensor)) else self.data[k][idx]
        if self.transform is not None: 
            input = self.transform(input)
        return input
        
    def _check_exists(self):  
        return os.path.exists(os.path.join(self.root, 'processed'))

    def download(self):
        if self._check_exists():
            return
        self.download_MOSI_data()


    def download_MOSI_data(self):
        sys.path.append("./CMU-MultimodalSDK")
        from mmsdk import mmdatasdk
        def myavg(intervals,features):
            return np.average(features,axis=0)
        dirname = os.path.dirname(self.root)
        makedir_exist_ok(dirname)
        if(not os.path.exists(os.path.join(self.root, 'cmumosi'))):
            makedir_exist_ok(os.path.join(self.root, 'cmumosi'))
            cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,os.path.join(self.root, 'cmumosi'))
        else:
            cmumosi_highlevel = mmdatasdk.mmdataset(os.path.join(self.root, 'cmumosi'))
        if(not os.path.exists(os.path.join(self.root, 'deployed'))):
            makedir_exist_ok(os.path.join(self.root, 'deployed'))               
            cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])
            cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,os.path.join(self.root, 'cmumosi'))
            cmumosi_highlevel.align('Opinion Segment Labels')
            deploy_files={x:x for x in cmumosi_highlevel.computational_sequences.keys()}
            cmumosi_highlevel.deploy(os.path.join(self.root, 'deployed'),deploy_files)
        aligned_cmumosi_highlevel = mmdatasdk.mmdataset(os.path.join(self.root, 'deployed'))
        self.train_keys = mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds.standard_train_fold
        self.validation_keys = mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds.standard_valid_fold
        self.test_keys =  mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds.standard_test_fold
        train_data = {k:[] for k in self.supported_feature_names}
        validation_data = {k:[] for k in self.supported_feature_names}
        test_data = {k:[] for k in self.supported_feature_names}
        for k in self.supported_feature_names:
            exec('{} = aligned_cmumosi_highlevel.computational_sequences[\'{}\'].data'.format(k,self.supported_feature_names[k]))
            exec('for m in {0}:\n'.format(k) +
                '   tmp_data = {0}[m][\'features\'][:]\n'.format(k) +
                '   tmp_data[tmp_data == -np.inf] = 0\n' +
                '   if(k==\'label\'):\n' +
                '       tmp_data = tmp_data.item(0)\n' +
                '   else:\n' +
                '       tmp_data = tmp_data.astype(np.float32)\n' +
                '   if(m[:m.index(\'[\')] in self.train_keys):\n' +
                '       train_data[\'{0}\'].append(tmp_data)\n'.format(k) +
                '   elif(m[:m.index(\'[\')] in self.validation_keys):\n' +
                '       validation_data[\'{0}\'].append(tmp_data)\n'.format(k) +
                '   elif(m[:m.index(\'[\')] in self.test_keys):\n' +
                '       test_data[\'{0}\'].append(tmp_data)\n'.format(k) +
                '   else:\n' +
                '       raise ValueError(\'key not found in folds\')')
        save(train_data,os.path.join(self.root, 'processed', 'train.pt'))
        save(validation_data,os.path.join(self.root, 'processed', 'validation.pt'))
        save(test_data,os.path.join(self.root, 'processed', 'test.pt'))        
        return

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
