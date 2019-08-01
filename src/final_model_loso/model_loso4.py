import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

'''
 one. spectrogram +lstm
'''
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
        self.padding_cnn =[(int((self.kernel_size_cnn[i][0]-1)/2),int((self.kernel_size_cnn[i][1]-1)/2)) for i in range(len(out_channels))]
        self.kernel_size_pool = kernel_size_pool
        self.padding_pool=[(int((self.kernel_size_pool[i][0]-1)/2),int((self.kernel_size_pool[i][1]-1)/2)) for i in range(len(out_channels))]
        self.stride_pool = stride_pool
# data shape
        self.nfft = nfft
        strideF = self.nfft//4
# for putting all cells together
        self._all_layers = []
        self.num_layers_cnn=len(out_channels)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            cell = LFLB(self.in_channels[i], self.out_channels[i], self.kernel_size_cnn[i], self.stride_cnn[i],
                        self.padding_cnn[i], self.padding_pool[i],self.kernel_size_pool[i], self.stride_pool[i], self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            strideF=self.cnn_shape(strideF,self.kernel_size_cnn[i][0],self.stride_cnn[i][0],self.padding_cnn[i][0],
                                    self.kernel_size_pool[i][0],self.stride_pool[i][0],self.padding_pool[i][0])

        self.strideF=strideF
    def forward(self, input):
        x = input.to(self.device)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            x = getattr(self, name)(x)
        out = torch.flatten(x,start_dim=1,end_dim=2)
        return out
    def dimension(self):
        return self.strideF*self.out_channels[-1]


class FTLSTM(nn.Module):
    def __init__(self,inputx_dim,hidden_dim,num_layers_ftlstm,device):
        super(FTLSTM,self).__init__()
        self._all_layers=[]
        self.device=device
        self.inputx_dim=inputx_dim
        self.hidden_dim=hidden_dim
        self.num_layers_ftlstm=num_layers_ftlstm
        self.dropout_rate=0
        self.bidirectional=False
        self.ftlstm = nn.LSTM(self.inputx_dim, self.hidden_dim, self.num_layers_ftlstm, batch_first=True,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional).to(self.device)
    def forward(self,inputx):
        inputx = inputx.to(self.device)
        out, hn = self.ftlstm(inputx)
        return out
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
        cell=FTLSTM(inputx_dim,hidden_dim,num_layers_ftlstm,device)
        setattr(self,"ftlstm",cell)
        
        self.device=device
        self.hidden_dim_lstm=200
        self.num_layers=2
        self.num_labels=4
        self.weight=nn.Parameter(torch.FloatTensor([weight]),requires_grad=False)
        self.classification_raw=nn.Linear(hidden_dim,self.num_labels).to(self.device)

    def forward(self,input_lstm,input1,input2,target,seq_length,train=True):
        input1=input1.to(self.device)
        input_lstm=input_lstm.to(self.device)
        target=target.to(self.device)
        seq_length=seq_length.to(self.device)
        
        inputx=getattr(self,"cnn")(input1)
        outT=getattr(self,"ftlstm")(inputx.permute(0,2,1)).permute(0,2,1)
        out=torch.mean(outT,dim=2)
        out = self.classification_raw(out)
        p = self.weight
        target_index = torch.argmax(target, dim=1).to(self.device)
        pred_index = torch.argmax(out, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out,dim=1))
        losses_batch=F.cross_entropy(out,torch.max(target,1)[1])
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        if train:
            return losses_batch,correct_batch
        return losses_batch, correct_batch, (target_index, pred_index)