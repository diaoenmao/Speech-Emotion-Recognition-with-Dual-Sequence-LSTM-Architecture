from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb
from sklearn.model_selection import train_test_split

def split_data(data):
    input_train, input_test, target_train, target_test, input_lstm_train, input_lstm_test, seq_length_train, seq_length_test, segement_labels_train, segement_labels_test= train_test_split(
        data['input'], data['target'], data["input_lstm"],data["seq_length"],data["segment_labels"], test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train,"input_lstm": input_lstm_train,"seq_length": seq_length_train,"segment_labels": segement_labels_train}
    test = {'input': input_test, 'target': target_test,"input_lstm": input_lstm_test,"seq_length": seq_length_test, "segment_labels": segement_labels_test}
    return train, test
def combine(name):
    with open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'rb') as out1:
        dict1=pickle.load(out1)
    with open('/scratch/speech/raw_audio_dataset/'+name+'_spectrogram_segmented_full.pkl', 'rb') as out2:
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

    dict3={"input_lstm":dict1["input"],"input":dict2["input"],"seq_length": dict1["seq_length"],"target": dict2["target"], "segment_labels": dict2["segment_labels"]}
    train1,test1=split_data(dict3)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_full.pkl', 'wb') as full:
        pickle.dump(dict3,full)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_train.pkl', 'wb') as train:
        pickle.dump(train1,train)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_test.pkl', 'wb') as test:
        pickle.dump(test1,test)

    print('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_full.pkl')


class IEMOCAP(Dataset):
    def __init__(self, name,train=True):
        if train:
            pickle_in = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_train.pkl', 'rb')
        else:
            pickle_in=open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_test.pkl', 'rb')
        data = pickle.load(pickle_in)
        self.seq_length = data["seq_length"]
        self.input_lstm= data["input_lstm"]
        self.input = data["input"]
        self.target = data["target"]
        self.segment_labels=data["segment_labels"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        #temp=[torch.unsqueeze(torch.from_numpy(i),dim=3).permute(2,0,1,3) for i in self.input[index]]
        temp=[torch.from_numpy(i).permute(1,0).float() for i in self.input[index]]
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': self.seq_length[index],
                  'input': temp,
                  'target': self.target[index],
                  'segment_labels': self.segment_labels[index]}
        return sample


def my_collate(batch):
    input_lstm = [i['input_lstm'] for i in batch]
    seq_length = torch.tensor([i['seq_length'] for i in batch])
    segment_labels=[i['segment_labels'] for i in batch]
    temp=[]
    for i in batch:
        temp+=i['input']
    temp=pad_sequence(temp,batch_first=True)
    temp1=[]
    low=0
    high=0
    for i in segment_labels:
        high+=len(i)
        temp1.append(temp[low:high,:,:])
        low=high
    temp1=pad_sequence(temp1,batch_first=True)
    input = torch.Tensor([i.permute(0,1,3,2) for i in temp1]).float()
    #input shape B*max(len(segment))*Freq*max(T)
    target = torch.from_numpy(np.array([i['target'] for i in batch]))
    return input_lstm,input,target,seq_length. segment_labels
if __name__=="__main__":
    combine("linear")
    combine("mel")