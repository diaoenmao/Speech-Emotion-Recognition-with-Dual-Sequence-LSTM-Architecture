import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

class LSTM_Audio(nn.Module):
    def __init__(self, hidden_dim_lstm, num_layers, device,dropout_rate=0 ,bidirectional=False):
        super(LSTM_Audio, self).__init__()
        self.device = device
        self.num_features = 39
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.num_features, self.hidden_dim_lstm, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)

    def forward(self, input):
        input = input.to(self.device)
        out, hn = self.lstm(input)
        return out

class LFLB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, padding_cnn, padding_pool,kernel_size_pool, stride_pool, device):
        super(LFLB, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn = stride_cnn
        self.padding_cnn = padding_cnn
        self.padding_pool=padding_pool
        self.kernel_size_pool = kernel_size_pool
        self.stride_pool = stride_pool

        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size_cnn, stride=self.stride_cnn, padding=self.padding_cnn).to(self.device)
        self.batch = nn.BatchNorm2d(self.out_channels)
        self.max_pool = nn.MaxPool2d(self.kernel_size_pool, stride=self.stride_pool,padding=self.padding_pool)
        self.relu = nn.ReLU()

    def forward(self,input):
        #pdb.set_trace()
        input=input.to(self.device)
        out=self.cnn(input)
        out=self.batch(out)
        out=self.relu(out)
        out=self.max_pool(out)
        return out

class SpectrogramModel(nn.Module):

    def valid_cnn(self,x,k):
        return torch.floor(x-k+1)

    def valid_max(self,x,k,s):
        return torch.floor((x-k)/s+1)

    def cnn_shape(self,x,kc,sc,pc,km,sm,pm):
        temp = int((x+2*pc-kc)/sc+1)
        temp=int((temp+2*pm-km)/sm+1)
        return temp

    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, kernel_size_pool, stride_pool,
                        hidden_dim, num_layers, dropout_rate, num_labels, batch_size,
                        hidden_dim_lstm,num_layers_lstm,device, nfft, bidirectional=False):
        super(SpectrogramModel, self).__init__()
        self.device = device
        self.in_channels = [in_channels]+out_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn = stride_cnn
        self.padding_cnn =[int((self.kernel_size_cnn[i]-1)/2) for i in range(len(out_channels))]
        self.kernel_size_pool = kernel_size_pool
        self.padding_pool=[int((self.kernel_size_pool[i]-1)/2) for i in range(len(out_channels))]
        self.stride_pool = stride_pool

# lstm
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional
        self.hidden_dim_lstm=hidden_dim_lstm

# data shape
        self.nfft = nfft
        strideF = self.nfft//4
        #strideF = 128

# for putting all cells together
        self._all_layers = []
        self.num_layers_cnn=len(out_channels)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            cell = LFLB(self.in_channels[i], self.out_channels[i], self.kernel_size_cnn[i], self.stride_cnn[i],
                        self.padding_cnn[i], self.padding_pool[i],self.kernel_size_pool[i], self.stride_pool[i], self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            strideF=self.cnn_shape(strideF,self.kernel_size_cnn[i],self.stride_cnn[i],self.padding_cnn[i],
                                    self.kernel_size_pool[i],self.stride_pool[i],self.padding_pool[i])

        self.lstm = nn.LSTM(self.out_channels[-1]*strideF, self.hidden_dim, self.num_layers, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)

    def forward(self, input):
        #pdb.set_trace()
        x = input.to(self.device)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            x = getattr(self, name)(x)
        out = torch.flatten(x,start_dim=1,end_dim=2).permute(0,2,1)
        out, hn = self.lstm(out)
        out = out.permute(0,2,1)
        return out

class MultiSpectrogramModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, kernel_size_pool, stride_pool,
                    hidden_dim, num_layers, dropout_rate, num_labels, batch_size,
                    hidden_dim_lstm,num_layers_lstm, device, nfft, bidirectional=False):
        super(MultiSpectrogramModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn = stride_cnn
        self.kernel_size_pool = kernel_size_pool
        self.stride_pool = stride_pool

# lstm
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers_lstm = num_layers_lstm
        self.device = device
        self.bidirectional = bidirectional
        self.num_directions = 1 + self.bidirectional

        self._all_layers = []
        self.num_branches = 2

        for i in range(self.num_branches):
            name = 'spec_cell{}'.format(i)
            cell = SpectrogramModel(self.in_channels, self.out_channels, self.kernel_size_cnn, self.stride_cnn, self.kernel_size_pool, self.stride_pool,
                                self.hidden_dim, self.num_layers, self.dropout_rate, self.num_labels, self.batch_size,
                                self.hidden_dim_lstm, self.num_layers_lstm, self.device, nfft[i], self.bidirectional)
            setattr(self, name, cell)
            self._all_layers.append(cell)

        self.LSTM_Audio=LSTM_Audio(self.hidden_dim_lstm,self.num_layers,self.device,bidirectional=False)
        self.classification_hand = nn.Linear(self.hidden_dim_lstm, self.num_labels).to(self.device)

        self.classification_raw = nn.Sequential(
                                nn.Linear(self.hidden_dim_lstm*self.num_directions*self.num_branches, self.hidden_dim_lstm*self.num_directions*self.num_branches//2),
                                nn.ReLU(),
                                nn.linear(self.hidden_dim_lstm*self.num_directions*self.num_branches//2,self.num_labels)).to(self.device)
        
        #self.classification_raw=nn.Linear(self.hidden_dim*self.num_directions*self.num_branches,self.num_labels).to(self.device)
        self.weight= nn.Parameter(torch.FloatTensor([0]),requires_grad=False)

    def forward(self, input_lstm, input1, input2, target, seq_length):
        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        #input3 = input3.to(self.device)
        target = target.to(self.device)
        #for i in range(self.num_branches):
        name = 'spec_cell{}'
        input1 = getattr(self, name.format("0"))(input1)
        #print("DONE WITH 1")
        input2 = getattr(self, name.format("1"))(input2)
        #print("DONE WITH 2")
        #input3 = getattr(self, name.format("2"))(input3)
        #print("DONE WITH 3")
        #pdb.set_trace()
        out_lstm = self.LSTM_Audio(input_lstm).permute(0,2,1)
        temp = [torch.unsqueeze(torch.mean(out_lstm[k,:,:int(s.item())],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm = torch.cat(temp,dim=0)
        out1 = torch.mean(input1, dim=2)
        out2 = torch.mean(input2, dim=2)
        #out3 = torch.mean(input3, dim=2)
        #out1 = self.classification_raw(out1)
        #out2 = self.classification_raw(out2)
        #out3 = self.classification_raw(out3)
        out = [out1, out2]
        out = torch.cat(out, dim=1)
        p = torch.exp(10*self.weight)/(1+torch.exp(10*self.weight))
        #p = 0.25
        out = self.classification_raw(out)
        out_lstm = self.classification_hand(out_lstm)
        out_final = p*out + (1-p)*out_lstm

        target_index = torch.argmax(target, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out_final,dim=1))
        losses_batch_raw=F.cross_entropy(out,torch.max(target,1)[1])
        losses_batch_hand=F.cross_entropy(out_lstm,torch.max(target,1)[1])
        losses_batch=p*losses_batch_raw+(1-p)*losses_batch_hand
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        return losses_batch, correct_batch
