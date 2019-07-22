import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class CNNcell(nn.Module):

class RNNcell(nn.Module):



class FTLSTMCell(nn.Module):
    def alignment(self,inputx, inputy, RNN=False):
        temp=[]
        for i in range(inputy.shape[1].item()):
            temp1=torch.mean(inputx[:,(2*i):(2*i+3)],dim=1)
            temp.append(temp1)
        inputx=temp
        return inputx, inputy

    def __init__(self,  input_dim,hidden_dim, dropout=0):
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
        return ch_pool, ch, cc

    def init_hidden(self, batch_size, hidden, shape):

        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device))
