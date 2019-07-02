import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb

class IEMOCAP(Dataset):
    def __init__(self, train=True):
        pickle_in = ''
        if train:
            pickle_in = open('/scratch/speech/raw_audio_dataset/spectrogram_train.pkl', 'rb')
        else:
            pickle_in = open('/scratch/speech/raw_audio_dataset/spectrogram_test.pkl', 'rb')
        data = pickle.load(pickle_in)
#        pdb.set_trace()
        self.input = data["input"]
        self.target = data["target"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        sample = {'input': self.input[index],
                  'target': self.target[index]}
        return sample

def my_collate(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdb.set_trace()
    input = torch.from_numpy((np.array([item['input'] for item in batch])).astype('float')).permute(0,3,1,2)
    # input = [x.cuda() for x in input]
    target = torch.from_numpy(np.array([item['target'] for item in batch]))
    # seq_length = [x[0] for x in seq_length]
    # seq_length = torch.from_numpy(np.array(seq_length))
    return input, target
