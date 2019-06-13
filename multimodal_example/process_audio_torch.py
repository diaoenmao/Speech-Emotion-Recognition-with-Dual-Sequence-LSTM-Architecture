import torch
import pickle
from torch.utils.data import Dataset, DataLoader

class IEMOCAP(Dataset):
    def __init__(self):
        pickle_in = open('/scratch/speech/dataset.pkl','rb')
        data = pickle.load(pickle_in)
        self.seq_len = data["seq_len"]
        self.input = data["input"]
        self.target = data["target"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        sample = {'input': torch.from_numpy(self.input[index]),
                    'target': torch.from_numpy(self.target[index]),
                    'seq_len': torch.from_numpy(self.seq_len[index])}
        return sample

dataset = IEMOCAP()
train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=0)
