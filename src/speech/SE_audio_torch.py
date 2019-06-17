import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import pdb

torch.manual_seed(1)


class GRUAudio(nn.Module):

    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels,batch_size):
        super(GRUAudio, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size

        self.gru = nn.GRU(self.num_features, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout_rate).to(self.device)
        self.classification = nn.Linear(self.hidden_dim * self.num_layers, self.num_labels).to(self.device)
#        self.softmax = nn.Softmax()

    def forward(self, input, target, train=True):
        input = input.to(self.device)
        target = target.to(self.device)
        hidden = torch.randn(2, self.batch_size, 200)
        hidden = hidden.to(self.device)
        out, hn = self.gru(input, hidden)
#        print(out, out.shape)
        #if train:
        #    hn, _ = pad_packed_sequence(hn, batch_first=True)
        hn=hn.permute([1,0,2])
#        pdb.set_trace()
        hn=hn.reshape(hn.shape[0],-1)
#        pdb.set_trace()
        out = self.classification(hn)
#        out = self.softmax(out)
#        pdb.set_trace()
        loss = F.cross_entropy(out, torch.max(target, 1)[1])
        return out, loss
