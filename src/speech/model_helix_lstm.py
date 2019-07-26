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
class HelixLstmCell(nn.Module):
    def __init__(self,  inputx_dim,inputy_dim,hidden_dim_x,hidden_dim_y,special2,dropout=0):
        # inputx, inputy should be one single time step, B*D
        super(HelixLstmCell, self).__init__()
        self.inputx_dim=inputx_dim
        self.inputy_dim=inputy_dim
        self.hidden_dim_x=hidden_dim_x
        self.hidden_dim_y=hidden_dim_y
        self.special2=special2
        if self.special2:
            self.Wfx=nn.Linear(self.inputx_dim+self.hidden_dim_x,self.hidden_dim_x,bias=True)
            self.Wfy=nn.Linear(self.inputy_dim+self.hidden_dim_y,self.hidden_dim_y,bias=True)
        self.Wix=nn.Linear(self.inputx_dim+self.hidden_dim_x,self.hidden_dim_x,bias=True)
        self.Wiy=nn.Linear(self.inputy_dim+self.hidden_dim_y,self.hidden_dim_y,bias=True)
        self.Wox=nn.Linear(self.inputx_dim+self.hidden_dim_x,self.hidden_dim_x,bias=True)
        self.Woy=nn.Linear(self.inputy_dim+self.hidden_dim_y,self.hidden_dim_y,bias=True)
        self.Wcx=nn.Linear(self.inputx_dim+self.hidden_dim_x,self.hidden_dim_x,bias=True)
        self.Wcy=nn.Linear(self.inputy_dim+self.hidden_dim_y,self.hidden_dim_y,bias=True)
        self.Wax=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim_x,self.hidden_dim_x,bias=True)
        self.Way=nn.Linear(self.inputx_dim+self.inputy_dim+self.hidden_dim_y,self.hidden_dim_y,bias=True)
        #self.batchx=nn.BatchNorm1d(num_features=self.hidden_dim_x)
        #self.batchy=nn.BatchNorm1d(num_features=self.hidden_dim_y)

        self.dropout=nn.Dropout(p=dropout, inplace=False)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x,y,h,Cx,Cy,flag):
        if flag=="x":
            #fx=torch.sigmoid(self.Wfx(torch.cat([x,h],dim=1)))
            ix=torch.sigmoid(self.Wix(torch.cat([x,h],dim=1)))
            C_x=torch.tanh(self.Wcx(torch.cat([x,h],dim=1)))
            ox=torch.sigmoid(self.Wox(torch.cat([x,h],dim=1)))
            ax=torch.sigmoid(self.Wax(torch.cat([x,y,h],dim=1)))
            if self.special2:
                fx=torch.sigmoid(self.Wfx(torch.cat([x,h],dim=1)))
                C=fx*Cx+ix*C_x+ax*Cy
            else:
                C=ix*C_x+ax*Cy
            h=ox*torch.tanh(C)
            #out=self.batchx(h)
        if flag=="y":
            #fy=torch.sigmoid(self.Wfy(torch.cat([y,h],dim=1)))
            iy=torch.sigmoid(self.Wiy(torch.cat([y,h],dim=1)))
            C_y=torch.tanh(self.Wcy(torch.cat([y,h],dim=1)))
            oy=torch.sigmoid(self.Woy(torch.cat([y,h],dim=1)))
            ay=torch.sigmoid(self.Way(torch.cat([y,x,h],dim=1)))
            if self.special2:
                fy=torch.sigmoid(self.Wfy(torch.cat([y,h],dim=1)))
                C=fy*Cy+iy*C_y+ay*Cx
            else:
                C=iy*C_y+ay*Cx
            h=oy*torch.tanh(C)
            #out=self.batchy(h)

        return h,C

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim_x).to(self.device),
                torch.zeros(batch_size, self.hidden_dim_y).to(self.device),
                torch.zeros(batch_size, self.hidden_dim_x).to(self.device),
                torch.zeros(batch_size, self.hidden_dim_y).to(self.device))
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
        self.input_dims=[]
        self.time_dims=[]
        for i in range(self.num_branches):
            name = 'spec_cell{}'.format(i)
            cell = SpectrogramModel(self.in_channels, self.out_channels, self.kernel_size_cnn[i], self.stride_cnn[i], self.kernel_size_pool[i], self.stride_pool[i], self.device, nfft[i])
            setattr(self, name, cell)
            self.input_dims.append(getattr(self,name).dimension())
            self.time_dims.append(getattr(self,name).dimension_time())
            self._all_layers.append(cell)
        print("time scales after CNN:", self.time_dims)
    def sequence(self,input1,input2):
        sequence=[]
        x=input1.shape[2]//2
        y=input2.shape[2]
        while x>0 and y>0:
            sequence.append("y")
            sequence.append("x")
            sequence.append("x")
            y=y-1
            x=x-1
        if x==0 and y>0:
            for _ in range(y):
                sequence.append("y")
        if x>0 and y==0:
            for _ in range(x):
                sequence.append("x")
                sequence.append("x")
        return sequence
    def forward(self, input1, input2):
        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        name = 'spec_cell{}'
        input1 = getattr(self, name.format("0"))(input1)
        input2 = getattr(self, name.format("1"))(input2)
        sequence=self.sequence(input1,input2)
        return input1,input2,sequence
    def dimension(self):
        return self.input_dims[0],self.input_dims[1]
    def dimension_time(self):
        temp=sum(self.time_dims)
        return temp
class HelixLstm(nn.Module):
    def __init__(self,time,inputx_dim,inputy_dim,hidden_dim_x,hidden_dim_y,num_layers_helix,special2,device):
        super(HelixLstm,self).__init__()
        self._all_layers=[]
        self.device=device
        self.time=time
        self.inputx_dim=inputx_dim
        self.inputy_dim=inputy_dim
        self.hidden_dim_x=hidden_dim_x
        self.hidden_dim_y=hidden_dim_y
        self.num_layers_helix=num_layers_helix
        self.special2=special2
        for i in range(num_layers_helix):
            name = 'helixlstm_cell{}'.format(i)
            cell = HelixLstmCell(inputx_dim,inputy_dim,hidden_dim_x,hidden_dim_y,self.special2)
            setattr(self, name, cell)
            self._all_layers.append(cell)
    def forward(self,inputx,inputy,sequence):
        internal_state_x = []
        internal_state_y = []
        assert self.time==len(sequence)
        outputx = []
        outputy = []
        timex=0
        timey=0
        for t,flag in enumerate(sequence):
            if flag=="x":
                x=inputx[:,:,timex]
                y=inputy[:,:,timey]
                if timex<inputx.shape[2]-1:
                    timex+=1
            if flag=="y":
                y=inputy[:,:,timey]
                x=inputx[:,:,timex]
                if timey<inputy.shape[2]-1:
                    timey+=1
            for i in range(self.num_layers_helix):
                name = 'helixlstm_cell{}'.format(i)
                if t==0:
                    bsize,_=x.size()
                    (hx,hy,Cx,Cy)=getattr(self, name).init_hidden(bsize)
                    internal_state_x.append((hx,Cx))
                    internal_state_y.append((hy,Cy))
                if flag=="x":
                    (hx,Cx)=internal_state_x[i]
                    (hy,Cy)=internal_state_y[i]
                    hx,Cx=getattr(self,name)(x,y,hx,Cx,Cy,flag)
                    internal_state_x[i]=hx,Cx
                if flag=="y":
                    (hy,Cy)=internal_state_y[i]
                    (hx,Cx)=internal_state_x[i]
                    hy,Cy=getattr(self,name)(x,y,hy,Cx,Cy,flag)
                    internal_state_y[i]=hy,Cy
            if flag=="x":
                outputx.append(hx)
            if flag=="y":
                outputy.append(hy)
        return torch.stack(outputx,dim=2),torch.stack(outputy,dim=2)
class CNN_HelixLstm(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size_cnn, 
                    stride_cnn, kernel_size_pool, stride_pool,nfft,
                    hidden_dim_x,hidden_dim_y,num_layers_helix,weight,special,special2,
                    device):
        super(CNN_HelixLstm,self).__init__()
        self._all_layers=[]
        cell=MultiSpectrogramModel(in_channels, out_channels, kernel_size_cnn, stride_cnn,
                                     kernel_size_pool, stride_pool, device, nfft)
        setattr(self,"cnn_multi",cell)
        inputx_dim,inputy_dim=getattr(self,"cnn_multi").dimension()
        time=getattr(self,"cnn_multi").dimension_time()
        print("time step after sequencing:",time)
        cell=HelixLstm(time,inputx_dim,inputy_dim,hidden_dim_x,hidden_dim_y,num_layers_helix,special2,device)
        setattr(self,"helix",cell)
        self.device=device
        self.hidden_dim_lstm=200
        self.hidden_dim_x=hidden_dim_x
        self.hidden_dim_y=hidden_dim_y
        self.num_layers=2
        self.num_labels=4
        self.weight=nn.Parameter(torch.FloatTensor([weight]),requires_grad=False)
        self.LSTM_Audio=LSTM_Audio(self.hidden_dim_lstm,self.num_layers,self.device,bidirectional=False)
        self.classification_hand = nn.Linear(self.hidden_dim_lstm, self.num_labels).to(self.device)
        self.special=special
        if self.special=="attention":
            self.attn_x=nn.Linear(self.hidden_dim_x,1).to(self.device)
            self.attn_y=nn.Linear(self.hidden_dim_y,1).to(self.device)
            self.classification_raw=nn.Linear(self.hidden_dim_x+hidden_dim_y,self.num_labels).to(self.device)
        elif self.special=="concat":
            self.classification_raw=nn.Linear(self.hidden_dim_x+hidden_dim_y,self.num_labels).to(self.device)
        else:
            self.classification_raw=nn.Linear(self.hidden_dim_x,self.num_labels).to(self.device)
    def forward(self,input_lstm,input1,input2,target,seq_length):
        input1=input1.to(self.device)
        input2=input2.to(self.device)
        input_lstm=input_lstm.to(self.device)
        target=target.to(self.device)
        seq_length=seq_length.to(self.device)
        inputx,inputy,sequence=getattr(self,"cnn_multi")(input1,input2)
        outx,outy=getattr(self,"helix")(inputx,inputy,sequence)
        out_lstm = self.LSTM_Audio(input_lstm).permute(0,2,1)
        temp = [torch.unsqueeze(torch.mean(out_lstm[k,:,:int(s.item())],dim=1),dim=0) for k,s in enumerate(seq_length)]
        out_lstm = torch.cat(temp,dim=0)
        out_lstm = self.classification_hand(out_lstm)
        if self.special=="attention":
            alpha_x=torch.unsqueeze(F.softmax(torch.squeeze(self.attn_x(outx.permute(0,2,1)),dim=2),dim=1),dim=2)
            alpha_y=torch.unsqueeze(F.softmax(torch.squeeze(self.attn_y(outy.permute(0,2,1)),dim=2),dim=1),dim=2)
            out=torch.cat([torch.squeeze(torch.bmm(outx,alpha_x),dim=2),torch.squeeze(torch.bmm(outy,alpha_y),dim=2)],dim=1)
        elif self.special=="concat":
            out=torch.cat([torch.mean(outx,dim=2),torch.mean(outy,dim=2)],dim=1)
        else:
            out=torch.mean(outx,dim=2)+torch.mean(outy,dim=2)
        out=self.classification_raw(out)
        p = self.weight
        out_final = p*out + (1-p)*out_lstm
        target_index = torch.argmax(target, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out_final,dim=1))
        losses_batch_raw=F.cross_entropy(out,torch.max(target,1)[1])
        losses_batch_hand=F.cross_entropy(out_lstm,torch.max(target,1)[1])
        losses_batch=p*losses_batch_raw+(1-p)*losses_batch_hand
        #losses_batch=F.cross_entropy(out,torch.max(target,1)[1]) 
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        return losses_batch, correct_batch