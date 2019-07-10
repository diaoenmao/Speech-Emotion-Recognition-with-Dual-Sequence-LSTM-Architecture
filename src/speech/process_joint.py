from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb
from sklearn.model_selection import train_test_split

def split_data(data):
    input_train, input_test, target_train, target_test, input_lstm_train, input_lstm_test, seq_length_train, seq_length_test= train_test_split(
        data['input'], data['target'], data["input_lstm"],data["seq_length"], test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train,"input_lstm": input_lstm_train,"seq_length": seq_length_train}
    test = {'input': input_test, 'target': target_test,"input_lstm": input_lstm_test,"seq_length": seq_length_test}
    return train, test
def combine():
    with open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'rb') as out1:
        dict1=pickle.load(out1)
    with open('/scratch/speech/raw_audio_dataset/spectrogram_segmented_dpi10_step40_full.pkl', 'rb') as out2:
        dict2=pickle.load(out2)
    flag=True
    for i in range(len(dict1["target"])):
        if np.argmax(dict1["target"][i])==np.argmax(dict2["target"][i]):
            continue
        else:
            flag=False
            break
    if flag: 
        print("Datasets consistent")
    else:
        raise ValueError("Datasets inconsistent")

    dict3={"input_lstm":dict1["input"],"input":dict2["input"],"seq_length": dict1["seq_length"],"target": dict2["target"]}
    train1,test1=split_data(dict3)
    with open('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_full.pkl', 'wb') as full:
        pickle.dump(dict3,full)
    with open('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_train.pkl', 'wb') as train:
        pickle.dump(train1,train)
    with open('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_test.pkl', 'wb') as test:
        pickle.dump(test1,test)

    print('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_train.pkl')


class IEMOCAP(Dataset):
    def __init__(self, train=True):
        if train:
            pickle_in = open('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_train.pkl', 'rb')
        else:
            pickle_in=open('/scratch/speech/hand_raw_dataset/IEMOCAP_39_FOUR_EMO_spectrogram_segmented_dpi10_step40_test.pkl', 'rb')
        data = pickle.load(pickle_in)
        self.seq_length = data["seq_length"]
        self.input_lstm= data["input_lstm"]
        self.input = data["input"]
        self.target = data["target"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        temp=[torch.unsqueeze(torch.from_numpy(i),dim=3).permute(2,0,1,3) for i in self.input[index]]
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': self.seq_length[index],
                  'input': torch.cat(temp,dim=3).float(),
                  'target': self.target[index]}
        return sample


def my_collate(batch):
    input_lstm = [i['input_lstm'] for i in batch]
    seq_length = torch.tensor([i['seq_length'] for i in batch])
    input = torch.cat([torch.unsqueeze(i['input'],dim=0) for i in batch],dim=0)
    target = torch.from_numpy(np.array([i['target'] for i in batch]))
    return input_lstm,input,target,seq_length
if __name__=='__main__':
    combine()


