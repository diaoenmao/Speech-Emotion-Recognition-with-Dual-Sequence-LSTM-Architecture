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
        self.classification = nn.Linear(self.hidden_dim, self.num_labels).to(self.device)

    def forward(self, input, target, train=True):
        input = input.to(self.device)
        target = target.to(self.device)
        hidden = torch.randn(2, self.batch_size, 200)
        hidden = hidden.to(self.device)
#        pdb.set_trace()
        out, hn = self.gru(input, hidden)
#        pdb.set_trace()
#        print(out, out.shape)
        if train:
            out, _ = pad_packed_sequence(out, batch_first=True)
#        pdb.set_trace()
        print("gru", out.shape)
        out = self.classification(out)
        print("linear", out.shape)
        loss = F.cross_entropy(out, target)
        return out, loss
