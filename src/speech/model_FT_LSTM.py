import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

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
    def __init__(self,  inputx_dim,inputy_dim,hidden_dim, dropout=0):
        # inputx, inputy should be one single time step, B*D
        super(FTLSTMCell, self).__init__()
        self.hidden_dim=hidden_dim
        self.inputx_dim=inputx_dim
        self.inputy_dim=inputy_dim
        self.WTf=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WFf=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WTi=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WFi=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WTo=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WFo=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WTc=nn.Linear(self.inputx_dim+self.hidden_dim,self.hidden_dim,bias=True)
        self.WFc=nn.Linear(self.inputy_dim+self.hidden_dim,self.hidden_dim,bias=True)

        self.batchF=nn.BatchNorm1d(num_features=self.hidden_dim)
        self.batchT=nn.BatchNorm1d(num_features=self.hidden_dim)
        self.dropout=nn.Dropout(p=dropout, inplace=False)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x,y,hT,hF,CT,CF):
        xyh=torch.cat([x,y,h],dim=1)
        fT=torch.sigmoid(self.WTf(torch.cat([x,y,hT],dim=1)))
        fF=torch.sigmoid(self.WFf(torch.cat([x,y,hF],dim=1)))
        iT=torch.sigmoid(self.WTi(torch.cat([x,y,hT],dim=1)))
        iF=torch.sigmoid(self.WFi(torch.cat([x,y,hF],dim=1)))
        C_T=torch.tanh(self.WTc(torch.cat([x,h],dim=1)))
        C_F=torch.tanh(self.WFc(torch.cat([y,h],dim=1)))
        oT=torch.sigmoid(self.WTo(torch.cat([x,y,hT],dim=1)))
        oF=torch.sigmoid(self.WFo(torch.cat([x,y,hF],dim=1)))
        CT=fT*CT+iT*C_T
        CF=fF*CF+iF*C_F
        hT=oT*torch.tanh(CT)
        hF=oF*torch.tanh(CF)
        outT=self.batchT(hT)
        outF=self.batchF(hF)
        return outT,outF,hT,hF,CT,CF

    def init_hidden(self, batch_size):
        return (nn.Parameter(torch.zeros(batch_size, self.hidden_dim)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, self.hidden_dim)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, self.hidden_dim)).to(self.device),
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
    def forward(self, input):
        x = input.to(self.device)
        for i in range(self.num_layers_cnn):
            name = 'lflb_cell{}'.format(i)
            x = getattr(self, name)(x)
        out = torch.flatten(x,start_dim=1,end_dim=2)
        return out
class MultiSpectrogramModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_cnn, stride_cnn, kernel_size_pool, stride_pool, device, nfft):
        super(MultiSpectrogramModel, self).__init__()
        self.device=device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_cnn = kernel_size_cnn
        self.stride_cnn = stride_cnn
        self.kernel_size_pool = kernel_size_pool
        self.stride_pool = stride_pool
        self._all_layers = []
        self.num_branches = 2
        for i in range(self.num_branches):
            name = 'spec_cell{}'.format(i)
            cell = SpectrogramModel(self.in_channels, self.out_channels, self.kernel_size_cnn[i], self.stride_cnn[i], self.kernel_size_pool[i], self.stride_pool[i], self.device, nfft[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)
    def alignment(self,input1,input2):
        # input2 has less time steps
        temp=[]
        if (input1.shape[2]-1)<(input2.shape[2])*2:
            input2=input2[:,:,:(input1.shape[2]-1)//2]
        for i in range(input2.shape[2]):
            temp1=torch.mean(input1[:,:,(2*i):(2*i+3)],dim=2)
            temp.append(temp1)
        inputx=torch.stack(temp,dim=2)
        inputy=input2
        return inputx,inputy
    def forward(self, input1, input2):
        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        name = 'spec_cell{}'
        input1 = getattr(self, name.format("0"))(input1)
        input2 = getattr(self, name.format("1"))(input2)
        inputx,inputy=self.alignment(input1,input2)
        return inputx,inputy
class FTLSTM(nn.Module):
    def __init__(self,time,inputx_dim,inputy_dim,hidden_dim,num_layers_ftlstm,device):
        super(FTLSTM,self).__init__()
        self._all_layers=[]
        self.device=device
        self.time=time
        self.inputx_dim=inputx_dim
        self.inputy_dim=inputy_dim
        self.hidden_dim=hidden_dim
        self.num_layers_ftlstm=num_layers_ftlstm
        for i in range(num_layers_ftlstm):
            name = 'ftlstm_cell{}'.format(i)
            cell = FTLSTMCell(inputx_dim,inputy_dim,hidden_dim)
            setattr(self, name, cell)
            self._all_layers.append(cell)
    def forward(self,inputx,inputy):
        internal_state = []
        outputT = []
        outputF=[]
        pdb.set_trace()
        for t in range(self.time):
            x=inputx[:,:,t]
            y=inputy[:,:,t]
            for i in range(self.num_layers_ftlstm):
                name = 'ftlstm_cell{}'.format(i)
                if t==0:
                    bsize,_=x.size()
                    (hT,hF,CT,CF)=getattr(self, name).init_hidden(bsize)
                    internal_state.append((hT,hF,CT,CF))
                (hT,hF,CT,CF)=internal_state[i]
                outT,outF,hT,hF,CT,CF=getattr(self,name)(x,y,hT,hF,CT,CF)
                internal_state[i]=hT,hF,CT,CF
            outputT.append(outT)
            outputF.append(outF)
        return torch.stack(outputT,dim=2),torch.stack(outputF,dim=2)
class CNN_FTLSTM(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size_cnn, 
                    stride_cnn, kernel_size_pool, stride_pool,nfft,
                    time,inputx_dim,inputy_dim,hidden_dim,num_layers_ftlstm,
                    device):
        super(CNN_FTLSTM,self).__init__()
        self._all_layers=[]
        cell=MultiSpectrogramModel(in_channels, out_channels, kernel_size_cnn, stride_cnn,
                                     kernel_size_pool, stride_pool, device, nfft)
        setattr(self,"cnn_multi",cell)
        cell=FTLSTM(time,inputx_dim,inputy_dim,hidden_dim,num_layers_ftlstm,device)
        setattr(self,"ftlstm",cell)
        self.device=device
        self.hidden_dim_lstm=200
        self.num_layers=2
        self.num_labels=4
        self.weight=0.5
        self.LSTM_Audio=LSTM_Audio(self.hidden_dim_lstm,self.num_layers,self.device,bidirectional=False)
        self.classification_hand = nn.Linear(self.hidden_dim_lstm, self.num_labels).to(self.device)
        self.classification_raw=nn.Linear(hidden_dim*2,self.num_labels).to(self.device)
    def forward(self,input_lstm,input1,input2,target,seq_length):
        input1=input1.to(self.device)
        input2=input2.to(self.device)
        input_lstm=input_lstm.to(self.device)
        target=target.to(self.device)
        seq_length=seq_length.to(self.device)
        inputx,inputy=getattr(self,"cnn_multi")(input1,input2)
        outT,outF=getattr(self,"ftlstm")(inputx,inputy)
        out_lstm = self.LSTM_Audio(input_lstm).permute(0,2,1)
        temp = [torch.unsqueeze(torch.mean(out_lstm[k,:,:int(s.item())],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm = torch.cat(temp,dim=0)
        out=torch.mean(torch.cat([outT,outF],dim=1),dim=2)
        p = self.weight
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








