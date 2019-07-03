import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import pdb

class IEMOCAP(Dataset):
    def __init__(self, train=True, segment=False):
        pickle_in = ''
        if train and segment: 
            pickle_in=open("/scratch/speech/raw_audio_dataset/raw_audio_segmented_train_equal_lengths.pkl","rb")
        elif train and not segment: 
            pickle_in = open('/scratch/speech/raw_audio_dataset/raw_audio_train_equal_lengths.pkl', 'rb')
            #pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5_train.pkl','rb')
        elif not train and segment:
            pickle_in=open('/scratch/speech/raw_audio_dataset/raw_audio_segmented_test_equal_lengths.pkl',"rb")
        else:
            pickle_in = open('/scratch/speech/raw_audio_dataset/raw_audio_test_equal_lengths.pkl', 'rb')
            #pickle_in = open('/scratch/speech/IEMOCAP_dictionary_5_test.pkl','rb')
        data = pickle.load(pickle_in)
        if not segment:
            self.seq_length = data["seq_length"]
        else:
            self.seq_length=[len(d) for d in data["input"]]
            # for compatibility

#        pdb.set_trace()
        self.input = data["input"]
        self.target = data["target"]
        self.segment_labels=data["segment_labels"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        sample = {'input': self.input[index],
                  'target': self.target[index],
                  'seq_length': self.seq_length[index],
                  'segment_labels': self.segment_labels}
        return sample

def my_collate_train(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.from_numpy(np.array([item['input'] for item in batch])).to(device)
    # input = [x.cuda() for x in input]
    target = torch.from_numpy(np.array([item['target'] for item in batch]))
    seq_length = torch.from_numpy(np.array([item['seq_length'] for item in batch]))
    # seq_length = [x[0] for x in seq_length]
    # seq_length = torch.from_numpy(np.array(seq_length))
    return [input, target, seq_length]
def my_collate_test(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = [item['input'] for item in batch]
    # input = [x.cuda() for x in input]
    target = torch.from_numpy(np.array([item['target'] for item in batch]))
    seq_length = torch.from_numpy(np.array([item['seq_length'] for item in batch]))
    segment_labels=[item["segment_labels"] for item in batch]
    # seq_length = [x[0] for x in seq_length]
    # seq_length = torch.from_numpy(np.array(seq_length))
    return [input, target, seq_length,segment_labels]
