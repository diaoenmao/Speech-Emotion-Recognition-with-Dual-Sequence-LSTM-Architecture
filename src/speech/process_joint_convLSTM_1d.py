from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class IEMOCAP(Dataset):
    def __init__(self, name, nfft, train=True):
        if train:
            f = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_train.pkl'.format(512), 'rb')
        else:
            f = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_test.pkl'.format(512), 'rb')
        data = pickle.load(f)
        self.input_lstm = data["input_lstm"]
        self.target = data["target"]
        self.input = data['input']

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(self.input[index])).float(), dim=0), size=256, mode='nearest'), dim=0)
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': self.input_lstm[index].shape[0],
                  'input': input,
                  'target': self.target[index]}
        return sample


def my_collate(batch):
    input_lstm = []
    seq_length = []
    target = []
    input = []
    for i in batch:
        input_lstm.append(i['input_lstm'])
        seq_length.append(i['seq_length'])
        target.append(i['target'])
        input.append(i['input'].float())
    input = torch.stack(input, dim=0)
    seq_length = torch.Tensor(seq_length)
    target = torch.from_numpy(np.array(target))
    input_lstm = pad_sequence(sequences=input_lstm, batch_first=True)
    input = torch.unsqueeze(input, dim=1)
    return input_lstm, input, target, seq_length
