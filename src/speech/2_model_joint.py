import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

class LSTM_Audio(nn.Module):
    def __init__(self, hidden_dim, num_layers, device,dropout_rate=0 ,bidirectional=False):
        super(LSTM_Audio, self).__init__()
        self.device = device
        self.num_features = 39
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)

    def forward(self, input):
        input = input.to(self.device)
        out, hn = self.lstm(input)
        return out

class SpectrogramModel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, padding_cnn, kernel_size_pool, stride_pool, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, hidden_dim_lstm,num_layers_lstm,device, bidirectional=False):
        super(SpectrogramModel, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn = stride_cnn
        self.padding_cnn = padding_cnn

        self.kernel_size_pool = kernel_size_pool
        self.stride_pool = stride_pool

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional

        self.cnn1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch1 = nn.BatchNorm2d(self.out_channels)
        self.cnn2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch2 = nn.BatchNorm2d(self.out_channels)
        self.cnn3 = nn.Conv2d(self.out_channels, self.out_channels*2, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch3 = nn.BatchNorm2d(self.out_channels*2)
        self.cnn4 = nn.Conv2d(self.out_channels*2, self.out_channels*2, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch4 = nn.BatchNorm2d(self.out_channels*2)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(self.kernel_size_pool//2, stride=self.stride_pool//2)
        self.max_pool = nn.MaxPool2d(self.kernel_size_pool, stride=self.stride_pool)
        self.max_pool4 = nn.MaxPool2d(int(self.kernel_size_pool*5/4), stride=int(self.stride_pool*5/4))
        self.lstm = nn.LSTM(int(640/160) * int(480/160), self.hidden_dim, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.classification = nn.Linear(self.hidden_dim * self.num_directions, self.num_labels).to(self.device)

        self.LSTM_Audio=LSTM_Audio(hidden_dim,num_layers,self.device,bidirectional=True)

    def forward(self, input_lstm,input, target,seq_length):
        input = input.to(self.device)
        target = target.to(self.device)
        out = self.cnn1(input)
        #print(out.shape)
        out = self.batch1(out)
        #print(out.shape)
        out = self.relu(out)
        #print(out.shape)
        out = self.max_pool1(out)
        #print(out.shape)
        out = self.cnn2(out)
        #print(out.shape)
        out = self.batch2(out)
        #print(out.shape)
        out = self.relu(out)
        #print(out.shape)
        out = self.max_pool(out)
        #print(out.shape)
        out = self.cnn3(out)
        #print(out.shape)
        out = self.batch3(out)
        #print(out.shape)
        out = self.relu(out)
        #print(out.shape)
        out = self.max_pool(out)
        #print(out.shape)
        out = self.cnn4(out)
        #print(out.shape)
        out = self.batch4(out)
        #print(out.shape)
        out = self.relu(out)
        #print(out.shape)
        out = self.max_pool4(out)
        #print(out.shape)
        #out = torch.flatten(out, start_dim=2, end_dim=3)
        out = out.view(list(out.size())[0], list(out.size())[1], -1)
        #pdb.set_trace()
        out, hn = self.lstm(out)

        out=out.permute(0,2,1)

        out_lstm=self.LSTM_Audio(input_lstm)
#        print(out.shape)
        out=torch.mean(out,dim=2)
        temp=[torch.unsqueeze(torch.mean(out_lstm[k,:,:s],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm=torch.cat(temp,dim=0)
        p=torch.exp(10*self.weight)/(1+torch.exp(10*self.weight))
        out=torch.cat([p*out,(1-p)*out_lstm],dim=1)
        out=self.classification(out)
        target_index = torch.argmax(target, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out,dim=1))
        losses_batch=F.cross_entropy(out,torch.max(target,1)[1])


        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)


        return  losses_batch,correct_batch