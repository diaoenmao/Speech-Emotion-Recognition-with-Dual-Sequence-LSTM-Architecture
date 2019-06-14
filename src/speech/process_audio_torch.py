import torch
import pickle
from torch.utils.data import Dataset, DataLoader


class IEMOCAP(Dataset):
    def __init__(self, train=True):
        pickle_in = ''
        if train:
            pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5.pkl', 'rb')
        else:
            pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5.pkl', 'rb')
        data = pickle.load(pickle_in)
        self.seq_length = data["seq_length"]
        self.input = data["input"]
        self.target = data["target"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        sample = {'input': torch.from_numpy(self.input[index]).float(),
                  'target': self.target[index],
                  'seq_length': torch.from_numpy(self.seq_length[index]).float()}
        return sample


def my_collate(batch):
    input = [item['input'] for item in batch]
    target = [item['target'] for item in batch]
    seq_length = [item['seq_length'] for item in batch]
    return [input, target, seq_length]


#dataset = IEMOCAP()
#sample = dataset[10]
#train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=0)
