import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

torch.manual_seed(1)

class GRUAudio(nn.Module):

    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels):
        super(GRUAudio, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels

        self.gru = nn.GRU(self.num_features, self.hidden_dim, self.num_layers, batch_first=True, dropout = self.dropout_rate)
        self.classification = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, input, target):
        out = self.gru(input)
        out = self.classification(out)
        loss = F.cross_entropy(out, target)
        return out, loss
