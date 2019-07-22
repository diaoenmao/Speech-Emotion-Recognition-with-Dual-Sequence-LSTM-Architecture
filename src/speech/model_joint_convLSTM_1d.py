import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

output = []

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

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, kernel_size_pool, stride_pool,padding,padding_pool,device):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride=1
        self.padding=padding
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool
        self.padding_pool=padding_pool

        self.Wxi = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.max_pool = nn.MaxPool1d(self.kernel_size_pool, stride=self.stride_pool, padding=self.padding_pool)
        self.batch = nn.BatchNorm1d(self.hidden_channels)

        self.dropout=nn.Dropout(p=0.1, inplace=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        self.device=device

    def forward(self, x, h, c):
        pdb.set_trace()
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        ch_pool=self.batch(self.max_pool(ch))
        #ch_pool=self.dropout(ch_pool)
        return ch_pool, ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # kernel size is also a list, same length as hidden_channels
    def cnn_shape(self,x,kc,sc,pc,km,sm,pm):
        temp = int((x+2*pc-kc)/sc+1)
        temp=int((temp+2*pm-km)/sm+1)
        return temp
    def __init__(self, input_channels, hidden_channels, kernel_size, stride, kernel_size_pool,stride_pool,hidden_dim_lstm,num_layers_lstm,device):
        super(ConvLSTM, self).__init__()
        self.device= device
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride=stride
        self.padding = [int((k-1) / 2) for k in self.kernel_size]
        self.num_layers = len(hidden_channels)
        self._all_layers = []
        self.num_labels=4
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers_lstm = num_layers_lstm
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool
        self.padding_pool=[int((kp-1)/2) for kp in self.kernel_size_pool]
        strideF=128
        for i in range(self.num_layers+1):
            if i<self.num_layers:
                name = 'cell{}'.format(i)
                cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i],self.kernel_size_pool[i],self.stride_pool[i],self.padding[i],self.padding_pool[i],self.device)
                setattr(self, name, cell)
                self._all_layers.append(cell)
                strideF=self.cnn_shape(strideF, self.kernel_size[i],self.stride[i], self.padding[i],self.kernel_size_pool[i],self.stride_pool[i],self.padding_pool[i])
            else:
                name="lstm"
                cell=LSTM_Audio(self.hidden_dim_lstm,self.num_layers_lstm,device=self.device,bidirectional=False)
                setattr(self,name,cell)
                self._all_layers.append(cell)
        self.linear_dim=int(self.hidden_channels[-1]*strideF)
        self.classification_convlstm = nn.Linear(self.linear_dim, self.num_labels)
        self.classification_lstm=nn.Linear(self.hidden_dim_lstm,self.num_labels)
        self.weight= nn.Parameter(torch.tensor(0).float(),requires_grad=False)
    def forward(self, input_lstm,input,target,seq_length):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times.
        ##data process here
        step=input.shape[2]
        internal_state = []
        outputs = []
        for s in range(step):
            x=input[:,:,:,s]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if s == 0:
                    bsize, _,shape = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=shape)
                    internal_state.append((h, c))
                    print(c.shape)

                # do forward
                (h, c) = internal_state[i]
                x, new_h, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (new_h, new_c)
            outputs.append(torch.flatten(x,start_dim=1, end_dim=2))
        out=torch.stack(outputs,dim=2)
        # B*dF*T
        input_lstm=input_lstm.to(self.device)
        seq_length=seq_length.to(self.device)
        out_lstm=getattr(self,"lstm")(input_lstm)
        out_lstm=out_lstm.permute(0,2,1)
        out=torch.mean(out,dim=2)
        temp=[torch.unsqueeze(torch.mean(out_lstm[k,:,:int(s.item())],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm=torch.cat(temp,dim=0)
        p=torch.exp(10*self.weight)/(1+torch.exp(10*self.weight))
        out=self.classification_convlstm(out)
        out_lstm=self.classification_lstm(out_lstm)
        out_final=p*out+(1-p)*out_lstm
        target_index = torch.argmax(target, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out_final,dim=1))
        losses_batch_raw=F.cross_entropy(out,torch.max(target,1)[1])
        losses_batch_hand=F.cross_entropy(out_lstm,torch.max(target,1)[1])
        losses_batch=p*losses_batch_raw+(1-p)*losses_batch_hand

        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)

        return  losses_batch,correct_batch
