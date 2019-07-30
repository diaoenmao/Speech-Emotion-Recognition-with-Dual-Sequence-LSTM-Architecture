import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RawAudioModel(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, padding_cnn, kernel_size_pool, stride_pool, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(RawAudioModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.cnn1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch1 = nn.BatchNorm1d(self.out_channels)
        self.cnn2 = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch2 = nn.BatchNorm1d(self.out_channels)
        self.cnn3 = nn.Conv1d(self.out_channels, self.out_channels*2, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch3 = nn.BatchNorm1d(self.out_channels*2)
        self.cnn4 = nn.Conv1d(self.out_channels*2, self.out_channels*2, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch4 = nn.BatchNorm1d(self.out_channels*2)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(self.kernel_size_pool, stride=self.stride_pool)
        self.lstm = nn.LSTM(int(128000/256), self.hidden_dim, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.classification = nn.Linear(self.hidden_dim * self.num_directions, self.num_labels).to(self.device)

    def forward(self, input, target, seq_length, train=True):
        input = input.to(self.device)
        target = target.to(self.device)
        out = self.cnn1(input)
#        print(out.shape)
        out = self.batch1(out)
#        print(out.shape)
        out = self.relu(out)
#        print(out.shape)
        out = self.max_pool(out)
#        print(out.shape)
        out = self.cnn2(out)
#        print(out.shape)
        out = self.batch2(out)
#        print(out.shape)
        out = self.relu(out)
#        print(out.shape)
        out = self.max_pool(out)
#        print(out.shape)
        out = self.cnn3(out)
#        print(out.shape)
        out = self.batch3(out)
#        print(out.shape)
        out = self.relu(out)
#        print(out.shape)
        out = self.max_pool(out)
#        print(out.shape)
        out = self.cnn4(out)
#        print(out.shape)
        out = self.batch4(out)
#        print(out.shape)
        out = self.relu(out)
#        print(out.shape)
        out = self.max_pool(out)
#        print(out.shape)

        out, hn = self.lstm(out)
#        print(out.shape)
        out = torch.mean(out, dim=1)

        out = self.classification(out)

        loss = F.cross_entropy(out, torch.max(target, 1)[1])
        return out, loss
