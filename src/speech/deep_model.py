import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import pdb
import math
torch.manual_seed(1)


class GRUAudio(nn.Module):

    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(GRUAudio, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional

        self.gru = nn.GRU(self.num_features, self.hidden_dim, self.num_layers, batch_first=True,
                          dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        
        self.classification = nn.Linear(self.hidden_dim * self.num_layers * self.num_directions, self.num_labels).to(
            self.device)

    #        self.softmax = nn.Softmax()

    def forward(self, input, target, train=True, seq_length=False):
        input = input.to(self.device)
        target = target.to(self.device)
        hidden = torch.randn(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim)
        hidden = hidden.to(self.device)
        out, hn = self.gru(input, hidden)
        #        print(out, out.shape)
        # if train:
        #    hn, _ = pad_packed_sequence(hn, batch_first=True)
        hn = hn.permute([1, 0, 2])
        hn = hn.reshape(hn.shape[0], -1)
        #        pdb.set_trace()
        out = self.classification(hn)
        #        out = self.softmax(out)
        #        pdb.set_trace()
        loss = F.cross_entropy(out, torch.max(target, 1)[1])
        return out, loss


class AttGRU(nn.Module):

    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(AttGRU, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional

        self.u = nn.Parameter(torch.zeros((self.num_directions * self.hidden_dim)), requires_grad=True)

        self.gru = nn.GRU(self.num_features, self.hidden_dim, self.num_layers, batch_first=True,
                          dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.classification = nn.Linear(self.hidden_dim * self.num_directions, self.num_labels).to(self.device)

    def forward(self, input, target, train=True, seq_length=False):
        input = input.to(self.device)
        target = target.to(self.device)
        hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim)
        hidden = hidden.to(self.device)
        out, hn = self.gru(input, hidden)

        out, _ = pad_packed_sequence(out, batch_first=True)

        mask = []
        #        pdb.set_trace()
        for i in range(len(seq_length)):
            mask.append([0] * int(seq_length[i].item()) + [1] * int(out.shape[1] - seq_length[i].item()))
        mask = torch.ByteTensor(mask)
        mask = mask.to(self.device)

        x = torch.matmul(out, self.u)
        x = x.masked_fill_(mask, -1e18)
        alpha = F.softmax(x, dim=1)

        input_linear = torch.sum(torch.matmul(alpha, out), dim=1)
        out = self.classification(input_linear)

        loss = F.cross_entropy(out, torch.max(target, 1)[1])
        #        print(self.u[10])
        return out, loss


class MeanPool(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(MeanPool, self).__init__()
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

        self.gru = nn.GRU(self.num_features, self.hidden_dim, self.num_layers, batch_first=True,
                          dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
        self.classification = nn.Linear(self.hidden_dim * self.num_directions, self.num_labels).to(self.device)

    def forward(self, input, target, train=True, seq_length=False):
        input = input.to(self.device)
        target = target.to(self.device)
        hidden = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim)
        hidden = hidden.to(self.device)
        out, hn = self.gru(input, hidden)

        out, _ = pad_packed_sequence(out, batch_first=True)

        out = torch.mean(out, dim=1)

        #        pdb.set_trace()

        out = self.classification(out)

        loss = F.cross_entropy(out, torch.max(target, 1)[1])
        return out, loss


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
        out_1_normalized=self.batch1(out_1)
        out_2=self.fc2(out_1_normalized)
        out_2_normalized=self.batch2(out_2)

        
        loss = F.cross_entropy(out_2_normalized, torch.max(target, 1)[1])
#        print(self.u[10])
        return out_2, loss


class Mean_Pool_2(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout_rate, num_labels, batch_size, bidirectional=False):
        super(Mean_Pool_2, self).__init__()
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
        out_1_normalized=self.batch1(out_1)
        out_2=self.fc2(out_1_normalized)
        out_2_normalized=self.batch2(out_2)

        
        loss = F.cross_entropy(out_2_normalized, torch.max(target, 1)[1])
#        print(self.u[10])
        return out_2, loss
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, kernel_size_pool=8, stride_pool=4):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride=1
        self.padding = int((kernel_size-1) / 2)
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool

        self.Wxi = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)
        
        self.max_pool = nn.MaxPool1d(self.kernel_size_pool, stride=self.stride_pool)
        self.batch = nn.BatchNorm1d(self.hidden_channels)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        ch_pool=self.batch(self.max_pool(ch))
        return ch_pool, ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape)).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape)).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape)).cuda()

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape)).cuda(),
                nn.Parameter(torch.zeros(batch_size, hidden, shape)).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # kernel size is also a list, same length as hidden_channels
    def __init__(self, input_channels, hidden_channels, kernel_size, step):
        super(ConvLSTM, self).__init__()
        assert len(hidden_channels)==len(kernel_size), "size mismatch"
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        self.num_labels=4
        self.linear_dim=16*18
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classification = nn.Linear(self.linear_dim, self.num_labels)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input, target):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times. 
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[step]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, shape = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=shape)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_h, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (new_h, new_c)
            outputs.append(x)
        ## mean pooling and loss function
        out=[torch.unsqueeze(o, dim=3) for o in outputs]
        out=torch.flatten(torch.mean(torch.cat(out,dim=3),dim=3),start_dim=1)

        out = self.classification(out)

        loss = F.cross_entropy(out, torch.max(target, 1)[1].to(self.device))



        return torch.unsqueeze(out,dim=0), torch.unsqueeze(loss, dim=0)