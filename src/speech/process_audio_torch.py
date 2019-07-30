import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb

class IEMOCAP(Dataset):
    def __init__(self, train=True):
        pickle_in = ''
        if train:
            pickle_in = open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_train.pkl', 'rb')
            #pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5_train.pkl','rb')
        else:
            pickle_in = open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_test.pkl', 'rb')
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


def my_collate(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = [item['input'].to(device) for item in batch]
    # input = [x.cuda() for x in input]
    target = torch.from_numpy(np.array([item['target'] for item in batch]))
    seq_length = torch.from_numpy(np.array([item['seq_length'].item() for item in batch]))
    # seq_length = [x[0] for x in seq_length]
    # seq_length = torch.from_numpy(np.array(seq_length))
    return [input, target, seq_length]

# dataset = IEMOCAP()
# sample = dataset[10]
# train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=0)
