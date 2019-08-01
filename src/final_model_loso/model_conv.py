import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_cnn,kernel_size_pool, stride_pool,padding,padding_pool,device):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride=stride_cnn
        self.padding=padding
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool
        self.padding_pool=padding_pool

        self.Wxi = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.max_pool = nn.MaxPool1d(self.kernel_size_pool, stride=self.stride_pool, padding=self.padding_pool)
        #self.batch = nn.BatchNorm1d(self.out_channels)

        self.dropout=nn.Dropout(p=0.1, inplace=False)

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
        ch_pool=self.max_pool(ch)
        return ch_pool, ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device))


class ConvLSTM(nn.Module):
    # in_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # kernel size is also a list, same length as out_channels
    def cnn_shape(self,x,kc,sc,pc,km,sm,pm):
        temp = int((x+2*pc-kc)/sc+1)
        temp=int((temp+2*pm-km)/sm+1)
        return temp
    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, kernel_size_pool,stride_pool,nfft,device):
        super(ConvLSTM, self).__init__()
        self.device= device
        self.in_channels = [in_channels] + out_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn=stride_cnn
        self.num_layers = len(out_channels)
        self._all_layers = []
        self.num_labels=4
        self.hidden_dim_lstm = 200
        self.num_layers_lstm = 2
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool
        self.nfft = nfft
        strideF = self.nfft//4
        self.padding_cnn =[(int((self.kernel_size_cnn[0]-1)/2),int((self.kernel_size_cnn[1]-1)/2)) for i in range(self.num_layers)]
        self.padding_pool=[(int((self.kernel_size_pool[0]-1)/2),int((self.kernel_size_pool[1]-1)/2)) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.in_channels[i], self.out_channels[i], self.kernel_size_cnn[0],self.stride_cnn[0],self.kernel_size_pool[0],self.stride_pool[0],self.padding_cnn[i][0],self.padding_pool[i][0],self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            strideF=self.cnn_shape(strideF,self.kernel_size_cnn[0],self.stride_cnn[0],self.padding_cnn[i][0],
                                   self.kernel_size_pool[0],self.stride_pool,self.padding_pool[i][0])
        self.strideF=strideF
    def forward(self,input):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times.
        ##data process here
        step=input.shape[3]
        internal_state = []
        outputs = []
        for s in range(step):
            x=input[:,:,:,s]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if s == 0:
                    bsize, _,shape = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.out_channels[i],
                                                             shape=shape)
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_h, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (new_h, new_c)
            outputs.append(torch.flatten(x,start_dim=1, end_dim=2))
        out=torch.stack(outputs,dim=2)
        # B*dF*T
        return out
    def dimension(self):
        return self.strideF
class CNN_FTLSTM(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size_cnn, 
                    stride_cnn, kernel_size_pool, stride_pool,nfft,
                    hidden_dim,num_layers_ftlstm,weight,
                    device):
        super(CNN_FTLSTM,self).__init__()
        
        self._all_layers=[]
        cell=ConvLSTM(in_channels, out_channels, kernel_size_cnn[0], stride_cnn[0], kernel_size_pool[0],stride_pool[0],nfft[0],device)
        setattr(self,"cnn",cell)
        self.strideF=getattr(self,"cnn",cell).dimension()
        self.device=device
        self.hidden_dim_lstm=200
        self.num_layers=2
        self.num_labels=4
        self.weight=nn.Parameter(torch.FloatTensor([weight]),requires_grad=False)
        self.linear_dim=int(out_channels[-1]*self.strideF)
        self.classification_convlstm = nn.Linear(self.linear_dim, self.num_labels)
    def forward(self,input_lstm,input1,input2,target,seq_length,train=True):
        input1=input1.to(self.device)
        input2=input2.to(self.device)
        input_lstm=input_lstm.to(self.device)
        target=target.to(self.device)
        seq_length=seq_length.to(self.device)

        inputx=getattr(self,"cnn")(input1)
        inputx=torch.mean(inputx,dim=2)
        out1=self.classification_convlstm(inputx)
        p = self.weight
        out_final=out1
        target_index = torch.argmax(target, dim=1).to(self.device)
        pred_index = torch.argmax(out_final, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out_final,dim=1))
        losses_batch=F.cross_entropy(out_final,torch.max(target,1)[1])
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        if train:
            return losses_batch,correct_batch
        return losses_batch, correct_batch, (target_index, pred_index)
