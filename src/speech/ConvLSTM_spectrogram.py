import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np



class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, dropout=0.1, kernel_size_pool=8, stride_pool=4):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride=1
        self.padding = int((kernel_size-1) / 2)
        self.kernel_size_pool=kernel_size_pool
        self.stride_pool=stride_pool
        self.padding_pool=int((kernel_size_pool-1)/2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.max_pool = nn.MaxPool2d(self.kernel_size_pool, stride=self.stride_pool, padding=self.padding_pool)
        self.batch = nn.BatchNorm2d(self.hidden_channels)

        self.dropout=nn.Dropout(p=dropout, inplace=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape)).to(self.device)

        return (nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device),
                nn.Parameter(torch.zeros(batch_size, hidden, shape)).to(self.device))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # kernel size is also a list, same length as hidden_channels
    def __init__(self, input_channels, hidden_channels, kernel_size, step,attention_flag=False):
        super(ConvLSTM, self).__init__()
        assert len(hidden_channels)==len(kernel_size), "size mismatch"
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        self.num_labels=4

        
        # max pooling
        self.stride_pool1=(4,4)
        self.stride_pool2=(4,2)
        self.stride_pool3=(3,2)

        self.kernel_size_pool1=(8,8)
        self.kernel_size_pool2=(8,4)
        self.kernel_size_pool=(5,5)

        self.padding_pool1=(int((self.kernel_size_pool1[0]-1)/2),int((self.kernel_size_pool1[1]-1)/2))
        self.padding_pool2=(int((self.kernel_size_pool2[0]-1)/2),int((self.kernel_size_pool2[1]-1)/2))
        self.padding_pool3=(int((self.kernel_size_pool3[0]-1)/2),int((self.kernel_size_pool3[1]-1)/2))



        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear_dim=int(self.hidden_channels[-1]*640/(self.step*self.stride_pool1[1]*self.stride_pool2[1]*self.stride_pool3[1])*480/(self.stride_pool1[0]*self.stride_pool2[0]*self.stride_pool3[0]))
        self.classification = nn.Linear(self.linear_dim, self.num_labels)
        self.attention=nn.Parameter(torch.zeros(self.linear_dim))
        self.attention_flag=attention_flag
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)


    def forward(self, input, target, seq_length):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times.
        ##data process here
        input=input.float().to(self.device)
        input=torch.split(input,int(640/self.step),dim=2)
        internal_state = []
        outputs = []
        for step in range(self.step):
            x=input[step]
            for i in range(self.num_layers):
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, shape = x.size()
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
            out=torch.mean(torch.cat(out,dim=4))

        out=self.classification(out)
        target_index = torch.argmax(target, dim=1).to(self.device)
        temp=0
        temp1=0
        correct_batch=torch.tensor([0])
        losses_batch=0

        for i,j in enumerate(target_index):
            temp1+=seq_length[i].item()
            loss=-1.0*torch.sum(F.log_softmax(out[temp:temp1,:],dim=1)[:,j],dim=0)
            if j==torch.argmax(torch.sum(out[temp:temp1,:],dim=0)):
                correct_batch+=1
            temp=temp1
            losses_batch += loss
        losses_batch=losses_batch/length
        # losses_batch is normalized
        correct_batch=torch.unsqueeze(correct_batch,dim=0)
        losses_batch=torch.unsqueeze(losses_batch, dim=0)
        length=torch.unsqueeze(length,dim=0)

        return  losses_batch,correct_batch, length
