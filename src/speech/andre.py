import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import pdb
import math
torch.manual_seed(1)


class ATT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(ATT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional
        self.attn = nn.Linear(self.hidden_dim * self.num_directions, hidden_dim)
        self.u=nn.Parameter(torch.randn(self.hidden_dim))
        stdv = 1. / math.sqrt(self.u.shape[0])
        self.u.data.normal_(mean=0, std=stdv)
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.fc1 = nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim).to(self.device)
        self.batch1=nn.BatchNorm1d(self.hidden_dim)
        self.fc2=nn.Linear(self.hidden_dim,self.num_labels).to(self.device)

        self.batch2=nn.BatchNorm1d(self.num_labels)
        self.batchatt=nn.BatchNorm1d(self.hidden_dim * self.num_directions)

    def forward(self, input, target, seq_length, train=True):
        input = input.to(self.device)
        target = target.to(self.device)

        out, hn = self.lstm(input)

        out , _ =pad_packed_sequence(out,batch_first=True)

        mask=[]
#        pdb.set_trace()
        for i in range(len(seq_length)):
            mask.append([0]*int(seq_length[i].item())+[1]*int(out.shape[1]-seq_length[i].item()))
        mask=torch.ByteTensor(mask)
        mask=mask.to(self.device)

        out_att=torch.tanh(self.attn(out))

        x=torch.matmul(out_att,self.u)

        x=x.masked_fill_(mask,-1e18)

        alpha=F.softmax(x,dim=1)



        input_linear=torch.sum(torch.matmul(alpha,out),dim=1)
        input_linear_normalized=self.batchatt(input_linear)
        out_1 = self.fc1(input_linear_normalized)
        out_1_normalized=self.batch1(out1)
        out_2=self.fc2(out_1_normalized)
        out_2_normalized=self.batch2(out2)

        
        loss = F.cross_entropy(out_2_normalized, torch.max(target, 1)[1])
#        print(self.u[10])
        return out_final, loss
class Mean_Pool_2(nn.module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(ATT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional
        #self.attn = nn.Linear(self.hidden_dim * self.num_directions, hidden_dim)
        #self.u=nn.Parameter(torch.randn(self.hidden_dim))
        #stdv = 1. / math.sqrt(self.u.shape[0])
        #self.u.data.normal_(mean=0, std=stdv)
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.fc1 = nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim).to(self.device)
        self.batch1=nn.BatchNorm1d(self.hidden_dim)
        self.fc2=nn.Linear(self.hidden_dim,self.num_labels).to(self.device)

        self.batch2=nn.BatchNorm1d(self.num_labels)
        self.batchatt=nn.BatchNorm1d(self.hidden_dim * self.num_directions)

    def forward(self, input, target, seq_length, train=True):
        input = input.to(self.device)
        target = target.to(self.device)

        out, hn = self.lstm(input)

        out , _ =pad_packed_sequence(out,batch_first=True)

        x=torch.mean(out,dim=1)

        input_linear_normalized=self.batchatt(x)
        out_1 = self.fc1(input_linear_normalized)
        out_1_normalized=self.batch1(out1)
        out_2=self.fc2(out_1_normalized)
        out_2_normalized=self.batch2(out2)

        
        loss = F.cross_entropy(out_2_normalized, torch.max(target, 1)[1])
#        print(self.u[10])
        return out_final, loss

