import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np



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
        #pdb.set_trace()
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
    def __init__(self, input_channels, hidden_channels, kernel_size, kernel_size_pool,kernel_stride_pool,step,device,attention_flag=False):
        super(ConvLSTM, self).__init__()
        self.device= device

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        self.num_labels=4

        
        # max pooling
        self.kernel_size_pool=kernel_size_pool
        self.kernel_stride_pool=kernel_stride_pool


        strideF=1
        strideT=1
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i],self.kernel_size_pool[i],self.kernel_stride_pool[i],self.device)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            strideF*=self.kernel_stride_pool[i][0]
            strideT*=self.kernel_stride_pool[i][1]



        self.linear_dim=int(self.hidden_channels[-1]*(480/strideF)*(640/(self.step*strideT)))
        #self.linear_dim=480
        self.classification = nn.Linear(self.linear_dim, self.num_labels)

        self.attention=nn.Parameter(torch.zeros(self.linear_dim))
        self.attention_flag=attention_flag



    def forward(self, input, target):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times.
        ##data process here
        target=target.to(self.device)
        input=torch.to(self.device)
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
        ## mean pooling and loss function
        out=[torch.unsqueeze(o, dim=4) for o in outputs]
        out=torch.flatten(torch.cat(out,dim=4),start_dim=1,end_dim=3)
        # out.shape batch*kf1f2*T
        if self.attention_flag:
            alpha=torch.unsqueeze(F.softmax(torch.matmul(self.attention,out),dim=1),dim=2)
            out=torch.squeeze(torch.bmm(out,alpha),dim=2)
        else:
            out=torch.mean(out,dim=2)
        out=self.classification(out)
        target_index = torch.argmax(target, dim=1).to(self.device)
        correct_batch=torch.sum(target_index==torch.argmax(out,dim=1))
        losses_batch=F.cross_entropy(out,torch.max(target,1)[1])


        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)


        return  losses_batch,correct_batch

