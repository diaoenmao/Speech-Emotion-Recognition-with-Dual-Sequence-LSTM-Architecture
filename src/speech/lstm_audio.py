import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import pdb

class LSTM_Audio(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(LSTM_Audio, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional

        # self.u=nn.Parameter(torch.randn(self.num_directions*self.hidden_dim)).to(self.device)

        self.lstm = nn.LSTM(self.num_features, self.hidden_dim, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.classification = nn.Linear(self.hidden_dim * self.num_directions, self.num_labels).to(self.device)

    def forward(self, input, target, seq_length, train=True):
        input = input.to(self.device)
        target = target.to(self.device)
        #hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim)
        #hidden = hidden.to(self.device)
#        pdb.set_trace()
        out, hn = self.lstm(input)

        out, _ = pad_packed_sequence(out, batch_first=True)

        out = torch.mean(out, dim=1)

        #        pdb.set_trace()

        out = self.classification(out)

        loss = F.cross_entropy(out, torch.max(target, 1)[1])
        return out, loss
