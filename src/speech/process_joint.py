from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb

class IEMOCAP(Dataset):
    def __init__(self, train=True):
        if train:
            pickle_in_lstm = open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_train.pkl', 'rb')
            pickle_in = open('/scratch/speech/raw_audio_dataset/spectrogram_segmented_train.pkl', 'rb')
        else:
            pickle_in_lstm = open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_test.pkl', 'rb')
            pickle_in = open('/scratch/speech/raw_audio_dataset/spectrogram_segmented_test.pkl', 'rb')
        data_lstm = pickle.load(pickle_in_lstm)
        self.seq_length = data_lstm["seq_length"]
        self.input_lstm= data_lstm["input"]
        data = pickle.load(pickle_in)
        self.input = data["input"]
        self.target = data["target"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        temp=[torch.unsqueeze(torch.from_numpy(i),dim=3).permute(2,0,1,3) for i in self.input[index]]
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': torch.tensor(self.seq_length[index]),
                  'input': torch.cat(temp,dim=3).float(),
                  'target': self.target[index]}
        return sample


def my_collate(batch):
    input_lstm = [i['input_lstm'] for i in batch]
    seq_length = torch.from_numpy(np.array([i['seq_length'] for i in batch]))
    input_lstm = pad_sequence(sequences=input_lstm, batch_first=True)
    input_lstm = pack_padded_sequence(input_lstm, lengths=seq_length, batch_first=True, enforce_sorted=False)
    input = torch.from_numpy(np.array([i['input'] for i in batch]))
    target = torch.from_numpy(np.array([i['target'] for i in batch]))
    return input_lstm,input,target

