import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb

class IEMOCAP(Dataset):
    def __init__(self, train=True):
        pickle_in = ''
        if train:
            pickle_in = open('/scratch/speech/raw_audio_dataset/raw_audio_train_equal_lengths.pkl', 'rb')
            #pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5_train.pkl','rb')
        else:
            pickle_in = open('/scratch/speech/raw_audio_dataset/raw_audio_test_equal_lengths.pkl', 'rb')
            #pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5_test.pkl','rb')
        data = pickle.load(pickle_in)
        self.seq_length = data["seq_length"]
#        pdb.set_trace()
        self.input = data["input"]
        self.target = data["target"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        sample = {'input': torch.from_numpy(self.input[index]).float(),
                  'target': self.target[index],
                  'seq_length': torch.tensor(self.seq_length[index]).float()}
        return sample
