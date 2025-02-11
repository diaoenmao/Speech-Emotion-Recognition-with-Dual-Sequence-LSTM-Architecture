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
    def __init__(self, input_channels, hidden_channels, kernel_size, kernel_size_pool, kernel_stride_pool,device, dropout=0.1):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride=1
        self.padding = (int((kernel_size[0]-1) / 2),int((kernel_size[1]-1) / 2))
        self.kernel_size_pool=kernel_size_pool
        self.kernel_stride_pool=kernel_stride_pool
        self.padding_pool=(int((kernel_size_pool[0]-1)/2),int((kernel_size_pool[1]-1)/2))


        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.max_pool = nn.MaxPool2d(self.kernel_size_pool, stride=self.kernel_stride_pool, padding=self.padding_pool)
        self.batch = nn.BatchNorm2d(self.hidden_channels)

        self.dropout=nn.Dropout(p=dropout, inplace=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        self.device=device


    def forward(self, x, h, c):
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
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0],shape[1])).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0],shape[1])).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0],shape[1])).to(self.device)

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape[0],shape[1])).to(self.device),
                nn.Parameter(torch.zeros(batch_size, hidden, shape[0],shape[1])).to(self.device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # kernel size is also a list, same length as hidden_channels
    def __init__(self, input_channels, hidden_channels, kernel_size, kernel_size_pool,kernel_stride_pool,step,device,num_devices,hidden_dim_lstm,num_layers_lstm,attention_flag=False):
        super(ConvLSTM, self).__init__()
        self.device= device
        self.num_devices=num_devices
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        self.num_labels=4
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers_lstm = num_layers_lstm
        self.kernel_size_pool=kernel_size_pool
        self.kernel_stride_pool=kernel_stride_pool
        strideF=1
        strideT=1
        for i in range(self.num_layers+1):
            if i<self.num_layers:
                name = 'cell{}'.format(i)
                cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i],self.kernel_size_pool[i],self.kernel_stride_pool[i],self.device)
                setattr(self, name, cell)
                self._all_layers.append(cell)
                strideF*=self.kernel_stride_pool[i][0]
                strideT*=self.kernel_stride_pool[i][1]
            else:
                name="lstm"
                cell=LSTM_Audio(self.hidden_dim_lstm,self.num_layers_lstm,device=self.device,bidirectional=True)
                setattr(self,name,cell)
                self._all_layers.append(cell)



        self.linear_dim=int(self.hidden_channels[-1]*(48/strideF)*(64/strideT))
        self.classification_convlstm = nn.Linear(self.linear_dim, self.num_labels)
        self.classification_lstm=nn.Linear(self.hidden_dim_lstm*2,self.num_labels)

        self.attention=nn.Parameter(torch.zeros(self.linear_dim))
        self.attention_flag=attention_flag

        self.weight= nn.Parameter(torch.tensor(-0.1).float(),requires_grad=True)



    def forward(self, input_lstm,input,target,seq_length, train=True):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times.
        ##data process here
        internal_state = []
        outputs = []
        for step in range(self.step):
            x=input[:,:,:,:,step]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, shapeF,shapeT = x.size()
                    shape=(shapeF,shapeT)
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=shape)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_h, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (new_h, new_c)
            outputs.append(x)
        out=[torch.unsqueeze(o, dim=4) for o in outputs]
        out=torch.flatten(torch.cat(out,dim=4),start_dim=1,end_dim=3)

        input_lstm=input_lstm.to(self.device)
        seq_length=seq_length.to(self.device)
        out_lstm=getattr(self,"lstm")(input_lstm)
        out_lstm=out_lstm.permute(0,2,1)
        # out.shape batch*kf1f2*T

        if self.attention_flag:
            alpha=torch.unsqueeze(F.softmax(torch.matmul(self.attention,out),dim=1),dim=2)
            out=torch.squeeze(torch.bmm(out,alpha),dim=2)
        else:
            out=torch.mean(out,dim=2)
            temp=[torch.unsqueeze(torch.mean(out_lstm[k,:,:s],dim=1),dim=0) for k,s in enumerate(seq_length)]
            out_lstm=torch.cat(temp,dim=0)
        p=torch.exp(10*self.weight)/(1+torch.exp(10*self.weight))
        #out=torch.cat([p*out,(1-p)*out_lstm],dim=1)
        out=self.classification_convlstm(out)
        out_lstm=self.classification_lstm(out_lstm)
        out_final=p*out_lstm+(1-p)*out
        target_index = torch.argmax(target, dim=1).to(self.device)
        pred_index = torch.argmax(out_final, dim=1)
        correct_batch=torch.sum(target_index==pred_index)
        losses_batch=F.cross_entropy(out_final,torch.max(target,1)[1])

        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)

        if train:
            return  losses_batch,correct_batch
        return losses_batch,correct_batch, (target_index, pred_index)
