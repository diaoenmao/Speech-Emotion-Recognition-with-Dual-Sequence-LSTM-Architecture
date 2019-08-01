import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
class SeparatedBatchNorm1d(nn.Module):
    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1):
        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.FloatTensor(num_features))
        self.bias = nn.Parameter(torch.FloatTensor(num_features))
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()
    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()
    def forward(self, input_, time):
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return F.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)
class LSTM_Audio(nn.Module):
    def __init__(self, hidden_dim, num_layers, device,dropout_rate=0.5 ,bidirectional=False):
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
        input=input.to(self.device)
        out=self.cnn(input)
        out=self.batch(out)
        out=self.relu(out)
        out=self.max_pool(out)
        return out

class FTLSTMCell(nn.Module):
    def __init__(self,  inputx_dim,hidden_dim, max_length, dropout=0,last_layer=False):
        # inputx, inputy should be one single time step, B*D
        super(FTLSTMCell, self).__init__()
        self.max_length = max_length
        self.hidden_dim=hidden_dim
        self.inputx_dim=inputx_dim
        # BN parameters
        self.batch = SeparatedBatchNorm1d(num_features=4 * self.hidden_dim, max_length=max_length)
        self.batchhT = nn.BatchNorm1d(self.hidden_dim)

        self.W=nn.Linear(self.inputx_dim+self.hidden_dim,4*self.hidden_dim,bias=True)

        self.dropout=nn.Dropout(p=dropout, inplace=False)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()
        self.last_layer=last_layer
    def reset_parameters(self):
        self.batch.reset_parameters()
        self.batch.bias.data.fill_(0)
        self.batch.weight.data.fill_(0.1)
    def forward(self, x,hT,CT,time_step):
        gates=self.batch(torch.sigmoid(self.W(torch.cat([x,hT],dim=1))),time=time_step)
        fT, iT, C_T,oT= (gates[:,:self.hidden_dim],gates[:,self.hidden_dim:2*self.hidden_dim],
                                gates[:,2*self.hidden_dim:3*self.hidden_dim],gates[:,3*self.hidden_dim:4*self.hidden_dim])
        CT=fT*CT+iT*C_T
        hT=oT*torch.tanh(CT)
        if self.last_layer:
            outT=self.dropout(self.batchhT(hT))
        else:
            outT=self.batchhT(hT)
        return outT,hT,CT
    def init_hidden(self, batch_size):
        return (nn.Parameter(torch.zeros(batch_size, self.hidden_dim)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, self.hidden_dim)).to(self.device))
class SpectrogramModel(nn.Module):
    def cnn_shape(self,x,kc,sc,pc,km,sm,pm):
        temp = int((x+2*pc-kc)/sc+1)
        temp=int((temp+2*pm-km)/sm+1)
        return temp
    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, kernel_size_pool, stride_pool,device, nfft):
        super(SpectrogramModel, self).__init__()
        self.device = device
        self.in_channels = [in_channels]+out_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn = stride_cnn
        self.padding_cnn =[(int((self.kernel_size_cnn[0]-1)/2),int((self.kernel_size_cnn[1]-1)/2)) for i in range(len(out_channels))]
        self.kernel_size_pool = kernel_size_pool
        self.padding_pool=[(int((self.kernel_size_pool[0]-1)/2),int((self.kernel_size_pool[1]-1)/2)) for i in range(len(out_channels))]
        self.stride_pool = stride_pool
# data shape
        self.nfft = nfft
        strideF = self.nfft//4
        if self.nfft==512:
            time=230
        if self.nfft==1024:
            time=120
# for putting all cells together
        self._all_layers = []
        self.num_layers_cnn=len(out_channels)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            cell = LFLB(self.in_channels[i], self.out_channels[i], self.kernel_size_cnn, self.stride_cnn,
                        self.padding_cnn[i], self.padding_pool[i],self.kernel_size_pool, self.stride_pool, self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            strideF=self.cnn_shape(strideF,self.kernel_size_cnn[0],self.stride_cnn[0],self.padding_cnn[i][0],
                                    self.kernel_size_pool[0],self.stride_pool,self.padding_pool[i][0])
            time=self.cnn_shape(time,self.kernel_size_cnn[1],self.stride_cnn[1],self.padding_cnn[i][1],
                                    self.kernel_size_pool[1],self.stride_pool,self.padding_pool[i][1])
        self.strideF=strideF
        self.time=time
    def forward(self, input):
        x = input.to(self.device)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            x = getattr(self, name)(x)
        out = torch.flatten(x,start_dim=1,end_dim=2)
        return out
    def dimension(self):
        return self.strideF*self.out_channels[-1]
    def dimension_time(self):
        return self.time


class FTLSTM(nn.Module):
    def __init__(self,time,inputx_dim,hidden_dim,num_layers_ftlstm,device):
        super(FTLSTM,self).__init__()
        self._all_layers=[]
        self.device=device
        self.time=time
        self.inputx_dim=inputx_dim
        self.hidden_dim=hidden_dim
        self.num_layers_ftlstm=num_layers_ftlstm
        for i in range(num_layers_ftlstm):
            name = 'ftlstm_cell{}'.format(i)
            if i==num_layers_ftlstm-1:
                cell = FTLSTMCell(inputx_dim,hidden_dim,self.time,last_layer=True)
                setattr(self, name, cell)
            else:
                cell = FTLSTMCell(hidden_dim,hidden_dim,self.time)
                setattr(self, name, cell)
            self._all_layers.append(cell)
    def forward(self,inputx):
        internal_state = []
        outputT = []
        for t in range(self.time):
            x=inputx[:,:,t]
            for i in range(self.num_layers_ftlstm):
                name = 'ftlstm_cell{}'.format(i)
                if t==0:
                    bsize,_=x.size()
                    (hT,CT)=getattr(self, name).init_hidden(bsize)
                    internal_state.append((hT,CT))
                (hT,CT)=internal_state[i]
                x,hT,CT=getattr(self,name)(x,hT,CT,t)
                internal_state[i]=hT,CT
                print(i)
                print(x.shape)
            outputT.append(x)
        return torch.stack(outputT,dim=2)
class CNN_FTLSTM(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size_cnn, 
                    stride_cnn, kernel_size_pool, stride_pool,nfft,
                    hidden_dim,num_layers_ftlstm,weight,
                    device):
        super(CNN_FTLSTM,self).__init__()
        
        self._all_layers=[]
        cell=SpectrogramModel(in_channels, out_channels, kernel_size_cnn, stride_cnn, kernel_size_pool, stride_pool,device, nfft)
        setattr(self,"cnn",cell)
        inputx_dim=getattr(self,"cnn").dimension()
        time=getattr(self,"cnn").dimension_time()
        print("time step after alignment:",time)
        cell=FTLSTM(time,inputx_dim,hidden_dim,num_layers_ftlstm,device)
        setattr(self,"ftlstm",cell)
        
        self.device=device
        self.hidden_dim_lstm=200
        self.num_layers=2
        self.num_labels=4
        self.weight=nn.Parameter(torch.FloatTensor([weight]),requires_grad=False)
        self.LSTM_Audio=LSTM_Audio(self.hidden_dim_lstm,self.num_layers,self.device,bidirectional=False)
        self.classification_hand = nn.Linear(self.hidden_dim_lstm, self.num_labels).to(self.device)
        self.classification_raw=nn.Linear(hidden_dim,self.num_labels).to(self.device)

    def forward(self,input_lstm,input1,input2,target,seq_length,train=True):
        
        input1=input1.to(self.device)
        input_lstm=input_lstm.to(self.device)
        target=target.to(self.device)
        
        seq_length=seq_length.to(self.device)
        
        inputx=getattr(self,"cnn")(input1)
        outT=getattr(self,"ftlstm")(inputx)
        
        out_lstm = self.LSTM_Audio(input_lstm).permute(0,2,1)
        temp = [torch.unsqueeze(torch.mean(out_lstm[k,:,:int(s.item())],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm = torch.cat(temp,dim=0)
        
        out=torch.mean(outT,dim=2)
        out = self.classification_raw(out)
        
        out_lstm = self.classification_hand(out_lstm)
        p = self.weight
        out_final = p*out+(1-p)*out_lstm 

        target_index = torch.argmax(target, dim=1).to(self.device)
        pred_index = torch.argmax(out_final, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out_final,dim=1))
        losses_batch_hand=F.cross_entropy(out_lstm,torch.max(target,1)[1])
        losses_batch_raw=F.cross_entropy(out,torch.max(target,1)[1])
        losses_batch=p*losses_batch_raw+(1-p)*losses_batch_hand
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        if train:
            return losses_batch,correct_batch
        return losses_batch, correct_batch, (target_index, pred_index)