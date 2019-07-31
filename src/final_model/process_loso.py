from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class IEMOCAP(Dataset):
    def __init__(self, train=True, session):
        if train:
            f = open('/scratch/speech/final_dataset/EMO39_mel_spectrogram_train_{}.pkl'.format(session), 'rb')
        else:
            f = open('/scratch/speech/final_dataset/EMO39_mel_spectrogram_test_{}.pkl'.format(session), 'rb')
        data = pickle.load(f)
        self.input_lstm = data['input_lstm']
        self.target = data['target']
        self.input1 = data['input1']
        self.input2 = data['input2']

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, index):
        input1 = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(self.input1[index])).float(), dim=0), size=230, mode='nearest'), dim=0)
        input2 = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(self.input2[index])).float(), dim=0), size=120, mode='nearest'), dim=0)
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
