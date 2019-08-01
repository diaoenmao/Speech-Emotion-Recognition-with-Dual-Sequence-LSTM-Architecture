import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
class LSTM_Audio(nn.Module):
    def __init__(self, hidden_dim,num_layers,device,dropout_rate=0 ,bidirectional=False):
        super(LSTM_Audio, self).__init__()
        self.device = device
        self.num_features = 39
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
    def forward(self, input_lstm):
        input_lstm = input_lstm.to(self.device)
        out_lstm,_ = self.lstm(input_lstm)
        return out_lstm

class CNN_FTLSTM(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size_cnn,
                    stride_cnn, kernel_size_pool, stride_pool,nfft,
                    hidden_dim,num_layers_ftlstm,weight,
                    device):
        super(CNN_FTLSTM,self).__init__()
        self.device=device
        self.hidden_dim_lstm=200
        self.num_layers=2
        self.num_labels=4
        self._all_layers=[]
        self.weight=nn.Parameter(torch.FloatTensor([weight]),requires_grad=False)
        self.LSTM_Audio=LSTM_Audio(self.hidden_dim_lstm,self.num_layers,self.device)
        self.classification_hand = nn.Linear(self.hidden_dim_lstm, self.num_labels).to(self.device)
    def forward(self,input_lstm,input1,input2,target,seq_length,train=True):
        input_lstm=input_lstm.to(self.device)
        target=target.to(self.device)
        seq_length=seq_length.to(self.device)
        out_lstm = self.LSTM_Audio(input_lstm).permute(0,2,1)
        temp = [torch.unsqueeze(torch.mean(out_lstm[k,:,:int(s.item())],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm = torch.cat(temp,dim=0)
        out_lstm = self.classification_hand(out_lstm)
        p = self.weight
        out_final = out_lstm
        target_index = torch.argmax(target, dim=1).to(self.device)
        pred_index = torch.argmax(out_final, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out_final,dim=1))
        losses_batch=F.cross_entropy(out_final,torch.max(target,1)[1])
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        if train:
            return losses_batch,correct_batch
        return losses_batch, correct_batch, (target_index, pred_index)
