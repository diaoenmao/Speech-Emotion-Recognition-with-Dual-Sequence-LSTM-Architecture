import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



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

        self.Wxi = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whi = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxf = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Whf = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding, bias=False)

        self.Wxc = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)

        self.Wxo = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, self.stride,self.padding,  bias=True)
        self.Who = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, self.stride, self.padding, bias=False)
        
        self.max_pool = nn.MaxPool1d(self.kernel_size_pool, stride=self.stride_pool, padding=self.padding_pool)
        self.batch = nn.BatchNorm1d(self.hidden_channels)

        self.dropout=nn.Dropout(p=dropout, inplace=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        self.device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
        self.device= torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.linear_dim=int(self.hidden_channels[-1]*(32000/step)/(4**self.num_layers))
        self.classification = nn.Linear(self.linear_dim, self.num_labels)
        self.attention=nn.Parameter(torch.zeros(self.linear_dim))
        self.attention_flag=attention_flag
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            setattr(self, name, cell)
            self._all_layers.append(cell)


    def forward(self, input, target, seq_length, length,multi_gpu=False):
        # input should be a list of inputs, like a time stamp, maybe 1280 for 100 times.
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
        out=[torch.unsqueeze(o, dim=3) for o in outputs]
        out=torch.flatten(torch.cat(out,dim=3),start_dim=1,end_dim=2)
        if self.attention_flag:
            alpha=torch.unsqueeze(F.softmax(torch.matmul(self.attention,out),dim=1),dim=2)
            out=torch.squeeze(torch.bmm(out,alpha),dim=2)
        else:
            out=torch.mean(torch.cat(out,dim=3))

        out=self.classification(out)
        target_index = torch.argmax(target, dim=1).to(self.device)
        temp=0
        temp1=0
        correct_batch=0
        losses_batch=0
        for i,j in enumerate(target_index):
            temp1+=seq_length[i].item()
            loss=torch.sum(out[temp:temp1,j],dim=0)
            if j==torch.argmax(torch.sum(out[temp:temp1,:],dim=0)):
                correct_batch+=1
            temp=temp1
        losses_batch += loss
        if multi_gpu:
            loss = F.cross_entropy(out, torch.max(target, 1)[1].to(self.device))
            out=torch.unsqueeze(out,dim=0)
            loss=torch.unsqueeze(loss, dim=0)
        return out, losses_batch/length,correct_batch

