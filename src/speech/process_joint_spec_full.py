from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb
from sklearn.model_selection import train_test_split

def split_data(data):
    input_train, input_test, target_train, target_test, input_lstm_train, input_lstm_test, seq_length_train, seq_length_test = train_test_split(
        data['input'], data['target'], data["input_lstm"],data["seq_length"],test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train,"input_lstm": input_lstm_train,"seq_length": seq_length_train}
    test = {'input': input_test, 'target': target_test,"input_lstm": input_lstm_test,"seq_length": seq_length_test}
    return train, test

def combine(name, nfft):
    with open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'rb') as out1:
        dict1=pickle.load(out1)
    with open('/scratch/speech/raw_audio_dataset/'+name+'_spectrogram_nfft{}_full.pkl'.format(nfft), 'rb') as out2:
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
    #seq_length_time=[]
    #for i in dict2['input']:
        #eq_length_time.append([j.shape[1] for j in i])


    dict3={"input_lstm":dict1["input"],"input":dict2["input"],"seq_length": dict1["seq_length"],"target": dict2["target"]}
    train1,test1=split_data(dict3)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_full.pkl'.format(nfft), 'wb') as full:
        pickle.dump(dict3,full)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_train.pkl'.format(nfft), 'wb') as train:
        pickle.dump(train1,train)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_test.pkl'.format(nfft), 'wb') as test:
        pickle.dump(test1,test)

    print('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_full.pkl'.format(nfft))


class IEMOCAP(Dataset):
    def __init__(self, name, nfft, train=True):
        if train:
            pickle_in = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_train.pkl'.format(nfft), 'rb')
            pickle_temp=open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_test.pkl'.format(nfft),'rb')
        else:
            pickle_in=open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_test.pkl'.format(nfft), 'rb')
            pickle_temp=open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_train.pkl'.format(nfft),'rb')
        data = pickle.load(pickle_in)
        data_temp=pickle.load(pickle_temp)
        self.seq_length = data["seq_length"]
        self.input_lstm= data["input_lstm"]
        self.target = data["target"]
        #self.segment_labels=data["segment_labels"]
        #self.seq_length_time=data["seq_length_time"]
        temp = data["input"]+data_temp['input']
        temp1 = []
        for utterance in temp:
            temp1.append(torch.from_numpy(utterance).permute(1,0).float())
        temp=pad_sequence(temp1,batch_first=True)
        self.input=temp[:len(data['input'])]
        #temp1=pad_sequence(temp1,batch_first=True)
        #self.input = temp1.permute(0,1,3,2).float()


    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        #temp=[torch.unsqueeze(torch.from_numpy(i),dim=3).permute(2,0,1,3) for i in self.input[index]]
        #temp=[torch.from_numpy(i).permute(1,0).float() for i in self.input[index]]
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': self.seq_length[index],
                  'input': self.input[index],
                  'target': self.target[index]}
        return sample


def my_collate(batch):
    input_lstm=[]
    seq_length=[]
    target=[]
    input=[]
    for i in batch:
        input_lstm.append(i['input_lstm'])
        seq_length.append(i['seq_length'])
        target.append(i['target'])
        input.append(i['input'])
    #seq_length=torch.Tensor(seq_length)
    target=torch.from_numpy(np.array(target))
    input_lstm = pad_sequence(sequences=input_lstm,batch_first=True)
    #input=pad_sequence(input,batch_first=True)
    input = [torch.unsqueeze(i, dim=0) for i in input]
    input = torch.cat(input, dim=0)
    #pdb.set_trace()
    input = torch.unsqueeze(input, dim=1)
    input = input.permute(0,1,3,2).float()

    #input shape B*max(len(segment))*Freq*max(T)
    return input_lstm,input,target,seq_length

if __name__=="__main__":
    combine("linear", 256)
    combine("mel", 256)
    combine("linear", 512)
    combine("mel", 512)
    combine("linear", 1024)
    combine("mel", 1024)
    combine("linear", 2048)
    combine("mel", 2048)
