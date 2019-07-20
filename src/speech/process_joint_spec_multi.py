from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import pdb
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


def split_data(data):
    input_train, input_test, target_train, target_test, input_lstm_train, input_lstm_test = train_test_split(
        data['input'], data['target'], data["input_lstm"],test_size=0.2, random_state=42)
    train = {'input': input_train, 'target': target_train,"input_lstm": input_lstm_train}
    test = {'input': input_test, 'target': target_test,"input_lstm": input_lstm_test}
    return train, test

def combine(name, nfft):
    with open('/scratch/speech/datasets/IEMOCAP_39_FOUR_EMO_full.pkl', 'rb') as out1:
        dict1=pickle.load(out1)

    with open('/scratch/speech/raw_audio_dataset/'+name+'_spectrogram_nfft{}_updated_full.pkl'.format(nfft), 'rb') as out2:
        dict2=pickle.load(out2)


    #with open('/scratch/speech/raw_audio_dataset/spectrogram_full.pkl', 'rb') as out2:
        #dict2=pickle.load(out2)

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


    dict3={"input_lstm":dict1["input"],"input":dict2["input"],"target": dict2["target"]}
    train1,test1=split_data(dict3)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_full.pkl'.format(nfft), 'wb') as full:
        pickle.dump(dict3,full)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_train.pkl'.format(nfft), 'wb') as train:
        pickle.dump(train1,train)
    with open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_test.pkl'.format(nfft), 'wb') as test:
        pickle.dump(test1,test)

    print('Done.')


class IEMOCAP(Dataset):
    def __init__(self, name, nfft, train=True):
        if train:
            f1 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_train.pkl'.format(nfft[0]), 'rb')
            f2 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_train.pkl'.format(nfft[1]), 'rb')
            #f3 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_train.pkl'.format(nfft[2]), 'rb')
        else:
            f1 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_test.pkl'.format(nfft[0]), 'rb')
            f2 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_updated_test.pkl'.format(nfft[1]), 'rb')
            #f3 = open('/scratch/speech/hand_raw_dataset/EMO39_'+name+'_spectrogram_nfft{}_test.pkl'.format(nfft[2]), 'rb')
        data1 = pickle.load(f1)
        data2 = pickle.load(f2)
        #data3 = pickle.load(f3)
        self.input_lstm = data1["input_lstm"]
        self.target = data1["target"]
        self.input1 = data1['input']
        self.input2 = data2['input']

    def __len__(self):
        return len(self.input1)

    def __getitem__(self, index):
        input1 = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(input1[index])).float(), dim=0), size=230, mode='nearest'), dim=0)
        input2 = torch.squeeze(F.interpolate(torch.unsqueeze(torch.from_numpy(10*np.log10(input2[index])).float(), dim=0), size=120, mode='nearest'), dim=0)
        sample = {'input_lstm': torch.from_numpy(self.input_lstm[index]).float(),
                  'seq_length': self.input_lstm[index].shape[0],
                  'input1': input1,
                  'input2': input2,
                  #'input3': input[2],
                  #'input': torch.Tensor(self.input[index].permute(2,1,0)).float()
                  'target': self.target[index]}
                  #'seq_length_spec':self.input[index].shape[1]}
        return sample


def my_collate(batch):
    input_lstm = []
    seq_length = []
    target = []
    input1 = []
    input2 = []
    #input3 = []
    for i in batch:
        input_lstm.append(i['input_lstm'])
        seq_length.append(i['seq_length'])
        target.append(i['target'])
        input1.append(i['input1'].float())
        input2.append(i['input2'].float())
        #input3.append(i['input3'].float())
    input1 = torch.stack(input1, dim=0)
    input2 = torch.stack(input2, dim=0)
    #input3 = torch.stack(input3, dim=0)
    #seq_length_spec=torch.Tensor(seq_length_spec)
    seq_length = torch.Tensor(seq_length)
    target = torch.from_numpy(np.array(target))
    #input=pad_sequence(sequences=input,batch_first=True)
    input_lstm = pad_sequence(sequences=input_lstm, batch_first=True)
    input1 = torch.unsqueeze(input1, dim=1)
    input2 = torch.unsqueeze(input2, dim=1)
    #input3 = torch.unsqueeze(input3, dim=1)
    #input = input.permute(0,1,3,2)

    #input shape B*max(len(segment))*Freq*max(T)
    return input_lstm, input1, input2, target, seq_length

if __name__=="__main__":

    #combine("linear", 256)
    #combine("mel", 256)
    #combine("linear", 512)
    combine("mel", 512)
    #combine("linear", 1024)
    combine("mel", 1024)
    #combine("linear", 2048)
    #combine("mel", 2048)

    #combine("linear",256)
