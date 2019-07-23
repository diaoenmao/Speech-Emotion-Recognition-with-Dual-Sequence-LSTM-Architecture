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
            f1 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_train.pkl'.format(nfft[0]), 'rb')
            f2 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_train.pkl'.format(nfft[1]), 'rb')
        else:
            f1 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_test.pkl'.format(nfft[0]), 'rb')
            f2 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_test.pkl'.format(nfft[1]), 'rb')
        data1 = pickle.load(f1)
        data2 = pickle.load(f2)
        self.input_lstm = data1["input_lstm"]
        self.target = data1["target"]
        self.input1 = data1['input']
        self.input2 = data2['input']

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, index):
        input1 = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(self.input1[index])).float(), dim=0), size=230, mode='nearest'), dim=0)
        input2 = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(self.input2[index])).float(), dim=0), size=230, mode='nearest'), dim=0)
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': self.input_lstm[index].shape[0],
                  'input1': input1,
                  'input2': input2,
                  'target': self.target[index]}
        return sample


def my_collate(batch):
    input_lstm = []
    seq_length = []
    target = []
    input1 = []
    input2 = []
    for i in batch:
        input_lstm.append(i['input_lstm'])
        seq_length.append(i['seq_length'])
        target.append(i['target'])
        input1.append(i['input1'].float())
        input2.append(i['input2'].float())
    input1 = torch.stack(input1, dim=0)
    input2 = torch.stack(input2, dim=0)
    seq_length = torch.Tensor(seq_length)
    target = torch.from_numpy(np.array(target))
    input_lstm = pad_sequence(sequences=input_lstm, batch_first=True)
    input1 = torch.unsqueeze(input1, dim=1)
    input2 = torch.unsqueeze(input2, dim=1)
    return input_lstm, input1, input2, target, seq_length

